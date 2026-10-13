"""
Core experiment pipeline.

Usage:
    from experiments.config import load_config
    from experiments.pipeline import run_pipeline
    run_pipeline(load_config())
"""

import importlib.util
import inspect
import io
import json
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.preprocessing.traditional import TraditionalDataPreprocessor
from data.preprocessing.llm import LLMDataPreprocessor
from data.preprocessing.prior import UNSW_NB15_PriorPreprocessor, CIC_IDS_2017_PriorPreprocessor, NSL_KDD_PriorPreprocessor, summarize_prior_preprocessing
from data.feature.feature_selection import FeatureSelectionDataPreprocessorV2
from data.feature.feature_extraction import FeatureExtractionDataPreprocessor
from data.feature.autoencoder_extraction import AutoEncoderExtractionDataPreprocessor
from models.base_model import BaseModel
from llm.llm_suggestions import LLM_Suggestions

from experiments.config import ExperimentConfig

_ROOT = Path(__file__).resolve().parents[1]
_GENERATED_DIR = _ROOT / "generated"
_HISTORY_FILE = _GENERATED_DIR / "history.json"
_RESULTS_DIR = _ROOT / "text_files"
_RANDOM_SEED = 42
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
        self.K_FEATURES = getattr(module, "K_FEATURES", None)
        # The LLM now also chooses the reduction kind and generates reduce_features() (Stage 3).
        self.REDUCTION_KIND = getattr(module, "REDUCTION_KIND", None)
        self.MUST_KEEP = getattr(module, "MUST_KEEP", None)
        self.MUST_DROP = getattr(module, "MUST_DROP", None)
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
        and config.classification_type == "multiclass"
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
    if config.dataset_name != "NSL_KDD" or config.classification_type != "multiclass":
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
    if config.classification_type == "binary":
        train_df_prior = prior.binary_prior_preprocessing(train_df.copy())
        test_df_prior = prior.binary_prior_preprocessing(test_df.copy()) if test_df is not None else None
    else:
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

    - traditional → TraditionalDataPreprocessor
    - USE_LLM=false → reuse GENERATED_FILE without an API call
    - USE_LLM=true → call the LLM (MODEL_ID), then load the newly generated file.

    If a new LLM call is made and `llm_usage` is provided, token-usage stats are
    written into it.
    """
    if config.llm_model == "traditional":
        return TraditionalDataPreprocessor(config.dataset_name)

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
    """Run preprocessing step, returns (X_train, y_train, X_test, y_test)."""
    is_trad = config.llm_model == "traditional"
    is_binary = config.classification_type == "binary"

    if is_trad:
        if is_binary:
            return preprocessor.binary_traditional_preprocessing(train_df, test_df, target_col=config.target_column)
        return preprocessor.multi_class_traditional_preprocessing(train_df, test_df, target_col=config.target_column)

    generated = getattr(preprocessor, "llm", None)
    if hasattr(generated, "select_resampling_candidate"):
        generated.select_resampling_candidate(resampling_candidate_id)

    if is_binary:
        return preprocessor.binary_LLM_preprocessing(train_df, test_df, target_col=config.target_column)
    return preprocessor.multi_class_LLM_preprocessing(train_df, test_df, target_col=config.target_column)


def _traditional_hybrid_feature_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    dataset_name: str,
    k: int,
    threshold: float = 0.95,
):
    """Rank by Stage-3 RF+MI evidence, then filter redundancy with Pearson."""
    fs = FeatureSelectionDataPreprocessorV2(dataset_name)
    hybrid_ranking = fs.compute_feature_target_correlation(
        X_train, y_train, method="rf_mi"
    )
    features = fs.feature_selection_preprocessing(
        hybrid_ranking,
        k,
        train_set=X_train,
        threshold=threshold,
        redundancy_method="pearson",
    )
    return X_train[features], X_test[features]


def _reduce_features(
    preprocessor,
    X_train,
    X_test,
    y_train,
    config: ExperimentConfig,
    k: int | None = None,
):
    """Apply feature reduction and return aligned reduced TRAIN/TEST frames.

    New generated reducers choose K internally from the fixed RF+MI threshold and
    expose a three-argument API. ``k`` remains only for the traditional path and
    legacy generated artifacts.
    """
    if config.llm_model == "traditional":
        if k is None:
            raise ValueError("The traditional reducer requires an explicit K.")
        selector = FeatureSelectionDataPreprocessorV2(config.dataset_name)
        ranking = selector.compute_feature_target_correlation(
            X_train, y_train, method="rf_mi"
        )
        features = _traditional_strict_pearson_prefix(
            ranking, X_train, int(k), threshold=0.90
        )
        return X_train[features], X_test[features]

    gen_reduce = getattr(getattr(preprocessor, "llm", None), "reduce_features", None)
    if callable(gen_reduce):
        parameters = inspect.signature(gen_reduce).parameters
        if len(parameters) == 3:
            return gen_reduce(X_train, X_test, y_train)
        if len(parameters) == 4 and k is not None:
            return gen_reduce(X_train, X_test, y_train, k)
        raise RuntimeError(
            "Generated reducer must implement adaptive "
            "reduce_features(X_train, X_test, y_train). Regenerate the artifact; "
            "legacy four-argument reducers require an explicit compatibility K."
        )

    if k is None:
        raise RuntimeError(
            "Adaptive feature selection requires a generated reduce_features() implementation."
        )
    return _traditional_hybrid_feature_selection(
        X_train, X_test, y_train, config.dataset_name, k
    )


def _base_model_type(classification_type: str) -> str:
    return "binary" if classification_type == "binary" else "multi_class"


def _resolve_k_values(preprocessor, config: ExperimentConfig):
    """Resolve K once so candidate selection and final evaluation use the same configuration."""
    llm_k = getattr(getattr(preprocessor, "llm", None), "K_FEATURES", None)
    if config.k_values:
        return list(config.k_values), f".env K_VALUES={list(config.k_values)}"
    if llm_k:
        return list(llm_k), f"LLM K_FEATURES={list(llm_k)}"
    return [10, 20, 30], "default=[10, 20, 30]"


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


def _evaluate_candidate_lightgbm(X_train, y_train, X_validation, y_validation):
    """Evaluate one resampling policy with the standard LightGBM configuration."""
    return BaseModel.evaluate_classifier(
        "LightGBM", X_train, y_train, X_validation, y_validation
    )


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


def _select_resampling_candidate(
    preprocessor,
    config: ExperimentConfig,
    inner_train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    candidates: dict,
    selection_k: int,
):
    """Evaluate policies without TEST access and return (winner_id, audit)."""
    records = []
    print("\nSelecting resampling candidate on TRAIN-derived validation...")
    print(
        f"  Inner TRAIN: {inner_train_df.shape} | Validation: {validation_df.shape} "
        f"| selection K={selection_k}"
    )

    for candidate_id, candidate in candidates.items():
        candidate_start = time.perf_counter()
        try:
            X_fit, y_fit, X_validation, y_validation = _preprocess(
                preprocessor,
                config,
                inner_train_df.copy(),
                validation_df.copy(),
                resampling_candidate_id=candidate_id,
            )
            X_fit = X_fit.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(drop=True)
            X_validation = (
                X_validation.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(drop=True)
            )
            y_fit = pd.Series(y_fit).reset_index(drop=True)
            y_validation = pd.Series(y_validation).reset_index(drop=True)

            runtime_k = min(int(selection_k), int(X_fit.shape[1]))
            if runtime_k < 1:
                raise RuntimeError("No feature remains for candidate selection.")
            X_fit_k, X_validation_k = _reduce_features(
                preprocessor, X_fit, X_validation, y_fit, config, runtime_k
            )
            metrics = _evaluate_candidate_lightgbm(
                X_fit_k, y_fit, X_validation_k, y_validation
            )
            elapsed = time.perf_counter() - candidate_start
            records.append({
                "candidate_id": candidate_id,
                "method": str(candidate.get("imbalance_handling", "None")),
                "status": "ok",
                "train_rows": int(len(y_fit)),
                "expansion_ratio": float(len(y_fit) / max(1, len(inner_train_df))),
                "runtime_seconds": float(elapsed),
                "selection_k": int(runtime_k),
                **metrics,
            })
        except Exception as exc:
            elapsed = time.perf_counter() - candidate_start
            records.append({
                "candidate_id": candidate_id,
                "method": str(candidate.get("imbalance_handling", "None")),
                "status": "failed",
                "runtime_seconds": float(elapsed),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  Candidate {candidate_id!r} failed safely: {type(exc).__name__}: {exc}")

    successful = [record for record in records if record["status"] == "ok"]
    none_record = next(
        (record for record in successful if record["candidate_id"] == "none"),
        None,
    )
    if none_record is None:
        raise RuntimeError("Mandatory resampling candidate 'none' failed; selection cannot continue.")

    comparison = pd.DataFrame(successful)
    if not comparison.empty:
        display_columns = [
            "candidate_id", "method", "weighted_f1", "accuracy",
            "train_rows", "expansion_ratio", "runtime_seconds", "selection_k",
        ]
        print(comparison[display_columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    ranked_candidates, ranking_audit = _rank_by_f1(successful)
    print(
        "[SELECTION RANKING] "
        f"successful={ranking_audit['successful_count']} | "
        f"weighted_f1_floor={ranking_audit['weighted_f1_floor']:.6f} | "
        f"near_ids={ranking_audit['weighted_f1_near_candidate_ids']}"
    )

    minimum_improvement = 0.0
    winner = ranked_candidates[0]
    decision_reason = (
        "selected by weighted-F1 with a strict < 0.003 near-tie band; candidates inside "
        "the band prefer lower resampling intensity (none < mild < moderate)."
    )

    print(
        f"Selected resampling candidate: {winner['candidate_id']} "
        f"({winner['method']}) — {decision_reason}\n"
    )
    return winner["candidate_id"], {
        "selection_source": "train_derived_validation",
        "selection_k": int(selection_k),
        "minimum_improvement_over_none": minimum_improvement,
        "weighted_f1_near_tolerance": ranking_audit["weighted_f1_near_tolerance"],
        "best_weighted_f1": ranking_audit["best_weighted_f1"],
        "weighted_f1_floor": ranking_audit["weighted_f1_floor"],
        "weighted_f1_near_candidate_ids": ranking_audit["weighted_f1_near_candidate_ids"],
        "ranking_metric_after_near_tie": ranking_audit["ranking_metric_after_near_tie"],
        "resampling_intensity_order": ranking_audit["resampling_intensity_order"],
        "winner_id": winner["candidate_id"],
        "winner_method": winner["method"],
        "decision_reason": decision_reason,
        "candidates": records,
    }


def _traditional_strict_pearson_prefix(
    ranking: pd.DataFrame,
    X_train: pd.DataFrame,
    k: int,
    threshold: float = 0.90,
) -> list[str]:
    """Return up to K ranked features without weakening the Pearson ceiling."""
    if k < 1:
        raise ValueError("K must be a positive integer.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Pearson threshold must be a fraction in [0, 1].")
    if not {"feature", "average_correlation"}.issubset(ranking.columns):
        raise ValueError("RF+MI ranking is missing required columns.")

    ranked_features = (
        ranking.sort_values(
            ["average_correlation", "feature"],
            ascending=[False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset="feature", keep="first")["feature"]
        .tolist()
    )
    missing = [feature for feature in ranked_features if feature not in X_train.columns]
    if missing:
        raise KeyError(f"Ranked features missing from inner TRAIN: {missing}")
    correlation = FeatureSelectionDataPreprocessorV2._feature_feature_correlation(
        X_train[ranked_features], method="pearson"
    )
    selected = []
    for feature in ranked_features:
        if selected and float(correlation.loc[feature, selected].max()) > threshold:
            continue
        selected.append(feature)
        if len(selected) == k:
            return selected

    return selected


def _select_traditional_k(
    validation_records: list[dict],
    weighted_f1_near_tolerance: float = _WEIGHTED_F1_NEAR_TOLERANCE,
) -> tuple[dict, list[dict]]:
    """Select the smallest K inside the strict weighted-F1 near-best band."""
    tolerance = float(weighted_f1_near_tolerance)
    if not np.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
        raise ValueError("weighted_f1_near_tolerance must be a finite fraction in [0, 1].")
    if not validation_records:
        raise RuntimeError("No traditional K candidate completed validation.")

    for record in validation_records:
        if not np.isfinite(float(record["weighted_f1"])):
            raise RuntimeError(
                f"K={record.get('k')} has non-finite weighted_f1."
            )

    best_weighted_f1 = max(
        float(record["weighted_f1"]) for record in validation_records
    )
    near_best = [
        record
        for record in validation_records
        if (
            best_weighted_f1 - float(record["weighted_f1"]) <= 0.0
            or best_weighted_f1 - float(record["weighted_f1"]) < tolerance
        )
    ]
    return min(near_best, key=lambda record: int(record["k"])), near_best


def run_pipeline(config: ExperimentConfig) -> dict:
    """Dispatch traditional compatibility runs or the LLM validation suite."""
    if config.llm_model == "traditional":
        return _run_pipeline_legacy(config)

    from experiments.validation_suite import run_validation_suite

    return run_validation_suite(config)


def _run_pipeline_legacy(config: ExperimentConfig) -> dict:
    """Select one traditional K on fixed TRAIN/VAL, then open TEST once."""
    from experiments.validation_suite import _load_or_create_fixed_validation_split

    config = _normalize_target_column(config)
    pipeline_start = time.perf_counter()
    pearson_threshold = 0.90
    k_near_tolerance = _WEIGHTED_F1_NEAR_TOLERANCE
    k_list = list(config.k_values) if config.k_values else [24, 26, 28, 30]
    k_list = [int(k) for k in k_list]
    if not k_list or len(k_list) != len(set(k_list)) or any(k < 1 for k in k_list):
        raise ValueError(
            "Traditional validation K values must be unique positive integers; "
            f"received {k_list!r}."
        )

    print(f"\n{'=' * 60}")
    print(f"Dataset       : {config.dataset_name}")
    print(f"Model ID      : {config.model_id}")
    print(f"Classification: {config.classification_type}")
    print(f"Target column : {config.target_column}")
    print(f"VAL K values  : {k_list}")
    print(f"Pearson limit : {pearson_threshold:.2f} (fixed; no relaxation)")
    print(f"{'=' * 60}\n")

    print("Loading official TRAIN; official TEST remains unopened...")
    train_df = pd.read_csv(config.train_path)
    split_source = _prepare_train_for_selection_split(config, train_df)
    inner_train, validation, split_manifest_path, split_manifest = (
        _load_or_create_fixed_validation_split(
            config, split_source, _rare_safe_train_validation_split
        )
    )
    print(
        f"  Inner TRAIN: {inner_train.shape} | VAL: {validation.shape} "
        "(fixed 80/20 split)\n"
    )

    preprocessor = build_preprocessor(config, train_df=inner_train, test_df=None)
    X_inner, y_inner, X_validation, y_validation = _preprocess(
        preprocessor, config, inner_train, validation
    )
    X_inner = (
        X_inner.replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index(drop=True)
    )
    y_inner = y_inner.reset_index(drop=True)
    X_validation = (
        X_validation.replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index(drop=True)
    )
    y_validation = y_validation.reset_index(drop=True)

    print("Ranking features once on inner TRAIN with RF+MI...")
    selector = FeatureSelectionDataPreprocessorV2(config.dataset_name)
    ranking = selector.compute_feature_target_correlation(
        X_inner, y_inner, method="rf_mi"
    )
    max_k = max(k_list)
    max_selected_features = _traditional_strict_pearson_prefix(
        ranking, X_inner, max_k, threshold=pearson_threshold
    )
    candidate_specs = []
    seen_effective_k = set()
    for requested_k in sorted(k_list):
        features = list(max_selected_features[:requested_k])
        effective_k = len(features)
        if effective_k < 1:
            raise RuntimeError(
                "Pearson filtering left no feature for traditional validation."
            )
        if effective_k in seen_effective_k:
            print(
                f"  Requested K={requested_k} collapses to effective K={effective_k}; "
                "duplicate candidate skipped."
            )
            continue
        seen_effective_k.add(effective_k)
        candidate_specs.append(
            {
                "requested_k": requested_k,
                "effective_k": effective_k,
                "features": features,
            }
        )

    print("\nSelecting K on VAL with standard LightGBM...")
    validation_records = []
    for candidate in candidate_specs:
        requested_k = candidate["requested_k"]
        effective_k = candidate["effective_k"]
        features = candidate["features"]
        metrics = BaseModel.evaluate_classifier(
            "LightGBM",
            X_inner[features],
            y_inner,
            X_validation[features],
            y_validation,
        )
        record = {
            "requested_k": requested_k,
            "k": effective_k,
            "features": features,
            **metrics,
        }
        validation_records.append(record)
        k_display = (
            f"requested K={requested_k} -> effective K={effective_k}"
            if requested_k != effective_k
            else f"K={effective_k}"
        )
        print(
            f"  {k_display}: weighted_f1={metrics['weighted_f1']:.6f} | "
            f"accuracy={metrics['accuracy']:.6f} | "
            f"runtime={metrics['runtime_seconds']:.2f}s"
        )

    winner, near_best = _select_traditional_k(
        validation_records, k_near_tolerance
    )
    selected_k = int(winner["k"])
    selected_requested_k = int(winner["requested_k"])
    selected_features = list(winner["features"])
    near_best_k_values = [record["k"] for record in near_best]
    selected_k_display = (
        f"requested K={selected_requested_k} -> effective K={selected_k}"
        if selected_requested_k != selected_k
        else f"K={selected_k}"
    )
    print(
        f"Locked {selected_k_display} from VAL weighted-F1; "
        f"near-best effective K values={near_best_k_values}."
    )

    print("\nLoading full official TEST after K is locked...")
    test_df = pd.read_csv(config.test_path)
    test_preprocessor = build_preprocessor(config, train_df=inner_train, test_df=None)
    X_fit, y_fit, X_test, y_test = _preprocess(
        test_preprocessor, config, inner_train, test_df
    )
    X_fit = (
        X_fit.replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index(drop=True)
    )
    y_fit = y_fit.reset_index(drop=True)
    X_test = (
        X_test.replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index(drop=True)
    )
    y_test = y_test.reset_index(drop=True)

    missing_fit = [
        feature for feature in selected_features if feature not in X_fit.columns
    ]
    missing_test = [
        feature for feature in selected_features if feature not in X_test.columns
    ]
    if missing_fit or missing_test:
        raise RuntimeError(
            "Validation-locked traditional features are missing during TEST evaluation: "
            f"inner_train={missing_fit}, test={missing_test}."
        )

    print(
        f"  Final fit: inner TRAIN {X_fit.shape[0]} rows | "
        f"official TEST: {X_test.shape[0]} rows | locked K={selected_k}\n"
    )
    model = BaseModel(_base_model_type(config.classification_type))
    results_df = model.evaluate_all_models(
        X_fit[selected_features],
        y_fit,
        X_test[selected_features],
        y_test,
    )
    all_results = {f"K={selected_k}": results_df}
    total_elapsed = time.perf_counter() - pipeline_start

    validation_table = pd.DataFrame([
        {
            "Requested K": record["requested_k"],
            "Effective K": record["k"],
            "Accuracy": record["accuracy"],
            "Weighted Precision": record["weighted_precision"],
            "Weighted Recall": record["weighted_recall"],
            "Weighted F1": record["weighted_f1"],
            "Time (s)": record["runtime_seconds"],
        }
        for record in validation_records
    ])
    validation_meta = {
        "split_manifest_path": split_manifest_path,
        "inner_train_rows": len(inner_train),
        "validation_rows": len(validation),
        "test_rows": len(test_df),
        "k_candidates": k_list,
        "effective_k_candidates": [record["k"] for record in validation_records],
        "selected_requested_k": selected_requested_k,
        "selected_k": selected_k,
        "selected_features": selected_features,
        "pearson_threshold": pearson_threshold,
        "rf_weight": 0.60,
        "mi_weight": 0.40,
        "controller": "LightGBM",
        "weighted_f1_near_tolerance": k_near_tolerance,
        "near_best_k_values": near_best_k_values,
        "validation_table": validation_table,
        "split_manifest": split_manifest,
    }
    reduction_meta = {
        f"K={selected_k}": {
            "total": selected_k,
            "semantic_seed": None,
            "semantic_cap": None,
            "mi_added": None,
            "audit_source": "traditional_validation_locked_features",
        }
    }

    print(f"\nWall-clock time : {total_elapsed:.2f}s")
    print("API latency     : 0.00s (no LLM call this run)")
    print("Cost per run    : $0.000000 USD")
    _save_results(
        all_results,
        config,
        total_elapsed=total_elapsed,
        llm_usage={},
        reduction_label="rf+mi+pearson90_val_selected",
        reduction_meta=reduction_meta,
        validation_meta=validation_meta,
    )
    return all_results


def _save_results(
    all_results: dict,
    config: ExperimentConfig,
    total_elapsed: float | None = None,
    llm_usage: dict | None = None,
    reduction_label: str | None = None,
    reduction_meta: dict | None = None,
    must_keep: list | None = None,
    resampling_meta: dict | None = None,
    validation_meta: dict | None = None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = config.llm_model.replace("-", "_").replace(".", "_").replace("/", "_")
    reduction_label = reduction_label or "selection"
    filename = f"{config.dataset_name}_{safe_model}_{config.classification_type}_{reduction_label}_{timestamp}.txt"
    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / filename

    lines = [
        f"Dataset       : {config.dataset_name}",
        f"Model ID      : {config.model_id}",
    ]
    lines += [
        f"Classification: {config.classification_type}",
        f"Reduction     : {reduction_label}",
        f"Timestamp     : {timestamp}",
    ]
    if total_elapsed is not None:
        lines.append(f"Wall-clock    : {total_elapsed:.2f}s")
    if llm_usage:
        lines.append(f"API latency   : {llm_usage.get('api_latency_seconds', 0.0):.2f}s ({llm_usage.get('calls', 0)} call(s))")
        lines.append(
            f"LLM tokens    : prompt={llm_usage.get('prompt_tokens', 0)}, "
            f"completion={llm_usage.get('completion_tokens', 0)}, "
            f"total={llm_usage.get('total_tokens', 0)}"
        )
        lines.append(f"Cost per run  : ${llm_usage.get('cost_usd', 0.0):.6f} USD")
    else:
        lines.append("API latency   : 0.00s (no LLM call)")
        lines.append("Cost per run  : $0.000000 USD")
    if validation_meta:
        lines += [
            "Training scope : fixed inner TRAIN 80% (no full-TRAIN refit)",
            "Validation     : fixed TRAIN holdout 20%",
            "Official TEST  : opened once after K lock; full official TEST",
            f"Fixed split   : {validation_meta['split_manifest_path']}",
            f"Rows          : inner TRAIN={validation_meta['inner_train_rows']}, "
            f"VAL={validation_meta['validation_rows']}, "
            f"TEST={validation_meta['test_rows']}",
            "Requested K   : " + str(validation_meta.get("k_candidates")),
            "Effective K   : " + str(validation_meta.get("effective_k_candidates")),
            f"K controller  : {validation_meta['controller']} weighted-F1",
            f"K near-tie    : difference < "
            f"{validation_meta['weighted_f1_near_tolerance']:.3f}; prefer smaller K",
            f"Near-best K   : {validation_meta['near_best_k_values']}",
            "Selected req. : " + str(validation_meta.get("selected_requested_k")),
            "Selected K    : " + str(validation_meta.get("selected_k")) + " (effective)",
            f"RF+MI weights : {validation_meta['rf_weight']:.2f}/"
            f"{validation_meta['mi_weight']:.2f} (ranking; no score cutoff)",
            f"Pearson limit : {validation_meta['pearson_threshold']:.2f} "
            "(fixed; no relaxation; no hard-fill)",
            "Resampling     : None",
            f"Selected features ({validation_meta['selected_k']}): "
            + ", ".join(validation_meta["selected_features"]),
            "",
            "=== VALIDATION K SELECTION (LightGBM) ===",
            validation_meta["validation_table"].to_string(index=False),
            "",
        ]
    # Only show the module-level MUST_KEEP list when it is actually declared at module level.
    # The newer hybrid reducer builds MUST_KEEP dynamically inside reduce_features(), so this
    # list is empty there — the accurate per-K counts are written under each "=== K=… ===" block.
    if must_keep:
        lines.append(f"MUST_KEEP ({len(must_keep)}): {', '.join(must_keep)}")
    if resampling_meta:
        lines.append(
            f"Resampling winner: {resampling_meta.get('winner_id')} "
            f"({resampling_meta.get('winner_method')})"
        )
        lines.append(
            f"Selection source : {resampling_meta.get('selection_source')} | "
            f"K={resampling_meta.get('selection_k')} | "
            f"minimum improvement={resampling_meta.get('minimum_improvement_over_none')}"
        )
        lines.append(f"Selection reason : {resampling_meta.get('decision_reason')}")
        for record in resampling_meta.get("candidates", []):
            if record.get("status") == "ok":
                lines.append(
                    f"Candidate {record['candidate_id']} ({record['method']}): "
                    f"weighted_f1={record['weighted_f1']:.6f}, "
                    f"accuracy={record['accuracy']:.6f}, "
                    f"rows={record['train_rows']}, "
                    f"expansion={record['expansion_ratio']:.6f}, "
                    f"runtime={record['runtime_seconds']:.3f}s"
                )
            else:
                lines.append(
                    f"Candidate {record.get('candidate_id')} ({record.get('method')}): "
                    f"FAILED — {record.get('error')}"
                )
    lines.append("")

    reduction_meta = reduction_meta or {}
    for k_label, df in all_results.items():
        lines.append(f"=== {k_label} ===")
        meta = reduction_meta.get(k_label)
        if meta:
            if meta.get("audit_source") == "traditional_validation_locked_features":
                lines.append(
                    "Features kept : " + str(meta.get("total")) + " "
                    "(validation-locked RF+MI ranking; Pearson <= 0.90)"
                )
            elif meta.get("semantic_seed") is not None and meta.get("mi_added") is not None:
                lines.append(
                    f"Features kept : {meta['total']} "
                    f"(semantic_seed: {meta['semantic_seed']}, "
                    f"MI added: {meta['mi_added']})"
                )
            else:
                lines.append(
                    f"Features kept : {meta['total']} "
                    "(semantic_seed: unavailable, MI added: unavailable; regenerate Stage 3)"
                )
        lines.append(df.to_string(index=False))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Results saved] {out_path}")
