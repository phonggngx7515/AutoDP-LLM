"""Orchestration helpers for the LLM multiclass intrusion-detection pipeline."""

import importlib.util
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from data.preprocessing.llm import LLMDataPreprocessor
from data.preprocessing.prior import (
    UNSW_NB15_PriorPreprocessor,
    CIC_IDS_2017_PriorPreprocessor,
    NSL_KDD_PriorPreprocessor,
    summarize_prior_preprocessing,
)
from experiments.config import ExperimentConfig
from llm.llm_suggestions import LLM_Suggestions

_ROOT = Path(__file__).resolve().parents[1]
_GENERATED_DIR = _ROOT / "generated"
_WEIGHTED_F1_NEAR_TOLERANCE = 0.003
_NSL_MULTICLASS_TARGET = "attack_cat"


class _GeneratedPreprocessorWrapper:
    """Wraps a dynamically-loaded generated module so it satisfies the LLMDataPreprocessor interface."""

    def __init__(self, module, artifact_filename: str | None = None):
        self._module = module
        self.ARTIFACT_FILENAME = artifact_filename
        self.RESAMPLING_CANDIDATES = getattr(module, "RESAMPLING_CANDIDATES", None)
        self.RESAMPLING_SELECTION = getattr(module, "RESAMPLING_SELECTION", None)
        self._selected_resampling_candidate_id = None
        try:
            preprocess_signature = inspect.signature(module.preprocess_data)
            self._accepts_resampling_candidate = (
                "resampling_candidate_id" in preprocess_signature.parameters
            )
        except (TypeError, ValueError):
            self._accepts_resampling_candidate = False

        self.IMBALANCE_HANDLING_METHOD = getattr(module, "IMBALANCE_HANDLING_METHOD", "None")
        # The LLM now also chooses the reduction kind and generates reduce_features() (Stage 3).
        self.REDUCTION_KIND = getattr(module, "REDUCTION_KIND", None)
        self.reduce_features = getattr(module, "reduce_features", None)

    def select_resampling_candidate(self, candidate_id: str | None):
        """Select one approved generated policy without changing common preprocessing."""
        if not self._accepts_resampling_candidate:
            if candidate_id not in (None, "none"):
                raise ValueError(
                    "This legacy generated preprocessor does not accept resampling_candidate_id."
                )
            self._selected_resampling_candidate_id = None
            return

        candidates = self.RESAMPLING_CANDIDATES or {}
        if candidate_id not in candidates:
            raise ValueError(
                f"Unknown resampling candidate {candidate_id!r}; "
                f"approved IDs are {list(candidates)}."
            )
        self._selected_resampling_candidate_id = candidate_id
        self.IMBALANCE_HANDLING_METHOD = str(
            candidates[candidate_id].get("imbalance_handling", "None")
        )

    def preprocess_data(self, train_set, test_set, train_label):
        if self._accepts_resampling_candidate:
            candidate_id = self._selected_resampling_candidate_id or "none"
            return self._module.preprocess_data(
                train_set, test_set, train_label, candidate_id
            )
        return self._module.preprocess_data(train_set, test_set, train_label)


def _load_generated_module(filename: str):
    """Dynamically import a generated preprocessor file by name."""
    filepath = _GENERATED_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Generated preprocessor not found: {filepath}")

    spec = importlib.util.spec_from_file_location("_generated_preprocessor", filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_generated_preprocessor"] = module
    spec.loader.exec_module(module)
    return module


def _get_prior_preprocessor(dataset_name: str):
    if dataset_name == "UNSW_NB15":
        return UNSW_NB15_PriorPreprocessor
    elif dataset_name == "CIC_IDS_2017":
        return CIC_IDS_2017_PriorPreprocessor
    elif dataset_name == "NSL_KDD":
        return NSL_KDD_PriorPreprocessor
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _normalize_target_column(config: ExperimentConfig) -> ExperimentConfig:
    """Use the canonical target created by dataset-specific prior preprocessing."""
    if (
        config.dataset_name == "NSL_KDD"
        and config.target_column != _NSL_MULTICLASS_TARGET
    ):
        print(
            f"[Config] NSL_KDD multiclass target normalized from "
            f"{config.target_column!r} to {_NSL_MULTICLASS_TARGET!r}."
        )
        return replace(config, target_column=_NSL_MULTICLASS_TARGET)
    return config


def _prepare_train_for_selection_split(
    config: ExperimentConfig,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize derived targets that must exist before the TRAIN-only split.

    NSL-KDD stores raw attack names in ``label`` (or ``class``), while its canonical
    multiclass target is the five-group ``attack_cat`` produced by prior preprocessing.
    Attach that derived target to a raw-data copy so splitting is stratified by the five
    canonical groups; later preprocessing can still consume the original raw columns.
    """
    train_df = train_df.copy()
    # Preserve existing UNSW/NSL split-manifest hashes; CIC labels need header cleanup.
    if config.dataset_name == "CIC_IDS_2017":
        train_df.columns = train_df.columns.str.strip()
    if config.dataset_name != "NSL_KDD":
        return train_df

    prior_df = NSL_KDD_PriorPreprocessor.multi_class_prior_preprocessing(train_df.copy())
    if _NSL_MULTICLASS_TARGET not in prior_df.columns:
        raise KeyError(
            "NSL_KDD multiclass prior preprocessing did not create "
            f"{_NSL_MULTICLASS_TARGET!r}. Expected a raw 'label' or 'class' column."
        )

    split_df = train_df.copy()
    split_df[_NSL_MULTICLASS_TARGET] = prior_df[_NSL_MULTICLASS_TARGET].to_numpy()
    return split_df


def _generate_via_llm(
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    usage_out: dict,
    test_df: pd.DataFrame | None = None,
    generation_seed: int = 42,
) -> str:
    """Call the LLM to generate a fresh preprocessing function. Returns the saved filename.

    Token-usage stats from the LLM call(s) are written into `usage_out` for the pipeline
    to report at the end of the run.
    """
    label_col = config.target_column

    # Apply prior preprocessing so the LLM analyses the same data its generated function will receive
    prior = _get_prior_preprocessor(config.dataset_name)
    train_df_prior = prior.multi_class_prior_preprocessing(train_df.copy())
    test_df_prior = prior.multi_class_prior_preprocessing(test_df.copy()) if test_df is not None else None

    # Summarize what the (dataset-specific) prior stage changed, so the LLM does not redo it.
    # Auto-diff (raw vs prior-processed) + the preprocessor's optional PRIOR_NOTES for this task type.
    prior_notes_map = getattr(prior, "PRIOR_NOTES", {}) or {}
    prior_notes = (prior_notes_map.get(config.classification_type, [])
                   if isinstance(prior_notes_map, dict) else list(prior_notes_map))
    prior_summary = summarize_prior_preprocessing(train_df, train_df_prior, label_col, prior_notes)

    llm = LLM_Suggestions(model_name=config.model_id)
    try:
        saved_file = llm.generate(
            generation_seed=generation_seed,
            train_df=train_df_prior,
            label_col_name=label_col,
            dataset_name=config.dataset_name,
            classification_type=config.classification_type,
            test_df=test_df_prior,
            prior_summary=prior_summary,
        )
    finally:
        # Failed generation attempts still consume tokens and belong in audit.
        usage_out.update(llm.api_caller.usage_summary())
    return saved_file


def build_preprocessor(
    config: ExperimentConfig,
    train_df: pd.DataFrame | None = None,
    llm_usage: dict | None = None,
    test_df: pd.DataFrame | None = None,
    generation_seed: int = 42,
):
    """Return a preprocessor object based on config.

    - USE_LLM=false → reuse GENERATED_FILE without an API call
    - USE_LLM=true → call the LLM (MODEL_ID), then load the newly generated file.

    If a new LLM call is made and `llm_usage` is provided, token-usage stats are
    written into it.
    """
    if not config.use_llm:
        if not config.generated_file:
            raise ValueError("GENERATED_FILE must be set when USE_LLM=false")
        filename = config.generated_file
        print(f"[LLM] Reusing generated preprocessor (no API call): {filename}")
    else:
        if train_df is None:
            raise ValueError("train_df is required to generate a new preprocessor via LLM")
        print(f"[LLM] Generating new preprocessor for ({config.dataset_name}, {config.model_id}, "
              f"{config.classification_type})...")
        filename = _generate_via_llm(
            config,
            train_df,
            llm_usage if llm_usage is not None else {},
            test_df=test_df,
            generation_seed=generation_seed,
        )

    module = _load_generated_module(filename)
    return LLMDataPreprocessor(
        config.dataset_name,
        _GeneratedPreprocessorWrapper(module, artifact_filename=filename),
    )


def _preprocess(
    preprocessor,
    config: ExperimentConfig,
    train_df,
    test_df,
    resampling_candidate_id: str | None = None,
):
    """Apply generated multiclass preprocessing and return aligned features/labels."""
    generated = getattr(preprocessor, "llm", None)
    if hasattr(generated, "select_resampling_candidate"):
        generated.select_resampling_candidate(resampling_candidate_id)
    return preprocessor.multi_class_LLM_preprocessing(
        train_df, test_df, target_col=config.target_column
    )


def _reduce_features(preprocessor, X_train, X_test, y_train):
    """Apply the generated adaptive reducer; feature selection is owned by the artifact."""
    reducer = getattr(getattr(preprocessor, "llm", None), "reduce_features", None)
    if not callable(reducer):
        raise RuntimeError("Generated artifact must implement reduce_features(X_train, X_test, y_train).")
    if len(inspect.signature(reducer).parameters) != 3:
        raise RuntimeError(
            "Generated reducer must use the adaptive three-argument API. Regenerate this artifact."
        )
    return reducer(X_train, X_test, y_train)


def _rare_safe_train_validation_split(
    train_df: pd.DataFrame,
    target_column: str,
    validation_fraction: float,
    random_state: int,
):
    """Create a deterministic TRAIN-only holdout without removing any class from inner TRAIN.

    Per-class floor allocation approximates the natural TRAIN priors. Classes too small to
    contribute at least one validation row safely remain exclusively in inner TRAIN.
    """
    if target_column not in train_df.columns:
        raise KeyError(f"Target column {target_column!r} is missing from TRAIN.")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")

    working = train_df.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    validation_positions = []

    grouped = working.groupby(target_column, observed=True, dropna=False, sort=False)
    group_positions = []
    for _, group in grouped:
        positions = group.index.to_numpy(dtype=int, copy=True)
        rng.shuffle(positions)
        group_positions.append(positions)
        n_validation = min(
            len(positions) - 1,
            int(np.floor(len(positions) * float(validation_fraction))),
        )
        if n_validation > 0:
            validation_positions.extend(positions[:n_validation].tolist())

    # Extremely small datasets can yield an empty holdout. Reserve one row from the
    # largest splittable class while still leaving at least one member in inner TRAIN.
    if not validation_positions:
        splittable = [positions for positions in group_positions if len(positions) >= 2]
        if not splittable:
            raise ValueError(
                "A TRAIN-derived validation split is impossible because every class is a singleton."
            )
        largest = max(splittable, key=len)
        validation_positions.append(int(largest[0]))

    validation_mask = np.zeros(len(working), dtype=bool)
    validation_mask[np.asarray(validation_positions, dtype=int)] = True
    inner_train = working.loc[~validation_mask].reset_index(drop=True)
    validation = working.loc[validation_mask].reset_index(drop=True)

    original_classes = set(working[target_column].astype(str).unique())
    inner_classes = set(inner_train[target_column].astype(str).unique())
    if inner_classes != original_classes:
        raise RuntimeError("Rare-safe split removed at least one target class from inner TRAIN.")
    return inner_train, validation


def _generated_resampling_contract(preprocessor):
    generated = getattr(preprocessor, "llm", None)
    candidates = getattr(generated, "RESAMPLING_CANDIDATES", None)
    selection = getattr(generated, "RESAMPLING_SELECTION", None)
    if not isinstance(candidates, dict) or "none" not in candidates:
        return None, None
    if not isinstance(selection, dict):
        raise RuntimeError(
            "Generated RESAMPLING_CANDIDATES requires a matching RESAMPLING_SELECTION contract."
        )
    return candidates, selection


def _rank_by_f1(
    successful_records: list[dict],
    weighted_f1_near_tolerance: float = _WEIGHTED_F1_NEAR_TOLERANCE,
) -> tuple[list[dict], dict]:
    """Apply the weighted-F1 near band, then prefer lighter resampling."""
    near_tolerance = float(weighted_f1_near_tolerance)
    if not np.isfinite(near_tolerance) or not 0.0 <= near_tolerance <= 1.0:
        raise ValueError(
            "weighted_f1_near_tolerance must be a finite fraction in [0, 1]; "
            f"received {weighted_f1_near_tolerance!r}."
        )
    if not successful_records:
        raise RuntimeError("No successful resampling candidate is available for ranking.")

    for record in successful_records:
        metric_value = float(record["weighted_f1"])
        if not np.isfinite(metric_value):
            raise RuntimeError(
                f"Candidate {record.get('candidate_id')!r} has non-finite "
                f"weighted_f1: {metric_value!r}."
            )

    best_weighted_f1 = max(
        float(record["weighted_f1"]) for record in successful_records
    )
    weighted_f1_floor = max(0.0, best_weighted_f1 - near_tolerance)
    near_best_records = [
        record
        for record in successful_records
        if (
            best_weighted_f1 - float(record["weighted_f1"]) <= 0.0
            or best_weighted_f1 - float(record["weighted_f1"]) < near_tolerance
        )
    ]

    intensity_order = {"none": 0, "mild": 1, "moderate": 2}
    ranked_records = sorted(
        near_best_records,
        key=lambda record: (
            intensity_order.get(str(record.get("candidate_id", "")), 99),
            str(record.get("candidate_id", "")),
        ),
    )
    audit = {
        "successful_count": len(successful_records),
        "weighted_f1_near_tolerance": near_tolerance,
        "best_weighted_f1": best_weighted_f1,
        "weighted_f1_floor": weighted_f1_floor,
        "weighted_f1_near_candidate_ids": [
            str(record.get("candidate_id")) for record in near_best_records
        ],
        "weighted_f1_near_count": len(near_best_records),
        "weighted_f1_near_comparison": "difference_strictly_less_than_tolerance",
        "ranking_metric_after_near_tie": "lower_resampling_intensity",
        "resampling_intensity_order": ["none", "mild", "moderate"],
    }
    return ranked_records, audit


def run_pipeline(config: ExperimentConfig) -> dict:
    """Run the LLM multiclass validation and held-out TEST workflow."""
    from experiments.validation_suite import run_validation_suite

    return run_validation_suite(config)
