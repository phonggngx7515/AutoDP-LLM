"""Multi-artifact validation controller for generated preprocessing artifacts."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import ExperimentConfig
from models.base_model import BaseModel

_WEIGHTED_F1_NEAR_TOLERANCE = 0.003
_RANDOM_SEED = 42
_RESULTS_DIR = Path(__file__).resolve().parents[1] / "text_files"
_SPLIT_DIR = _RESULTS_DIR / "validation_splits"


def _clean_outputs(outputs):
    X_train, y_train, X_eval, y_eval = outputs
    train_attrs = deepcopy(getattr(X_train, "attrs", {}))
    eval_attrs = deepcopy(getattr(X_eval, "attrs", {}))
    X_train = (
        pd.DataFrame(X_train)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index(drop=True)
    )
    X_eval = (
        pd.DataFrame(X_eval)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index(drop=True)
    )
    # pd.DataFrame(existing_dataframe) does not preserve attrs. Stage 3 relies
    # on the Stage-2 encoding/log audit, so restore all artifact-owned metadata
    # after the host normalizes the feature matrices.
    X_train.attrs = train_attrs
    X_eval.attrs = eval_attrs
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_eval = pd.Series(y_eval).reset_index(drop=True)
    if len(X_train) != len(y_train) or len(X_eval) != len(y_eval):
        raise RuntimeError("Preprocessing returned misaligned feature and label rows.")
    if list(X_train.columns) != list(X_eval.columns):
        raise RuntimeError("Preprocessing returned different TRAIN and evaluation schemas.")
    if X_train.shape[1] < 1:
        raise RuntimeError("Preprocessing removed every feature.")
    return X_train, y_train, X_eval, y_eval


def _artifact_filename(preprocessor):
    generated = getattr(preprocessor, "llm", None)
    filename = getattr(generated, "ARTIFACT_FILENAME", None)
    if not filename:
        raise RuntimeError("Generated artifact filename is unavailable for audit/refit.")
    return str(filename)


def _selected_resampling_details(run):
    """Return the validation-locked resampling policy for one artifact."""
    candidate_id = str(run.get("selected_candidate", "unknown"))
    candidates = run.get("resampling_candidates", {})
    candidate = (
        candidates.get(candidate_id, {})
        if isinstance(candidates, dict)
        else {}
    )
    if not isinstance(candidate, dict):
        candidate = {}
    return {
        "candidate_id": candidate_id,
        "method": str(
            candidate.get(
                "imbalance_handling",
                run.get("resampling_family", "unknown"),
            )
        ),
        "alpha": candidate.get("alpha"),
        "under_fraction": candidate.get("under_fraction"),
        "target_class_policy": candidate.get("target_class_policy"),
        "method_params": deepcopy(candidate.get("method_params", {})),
        "max_expansion_ratio": candidate.get("max_expansion_ratio"),
        "max_resampled_rows": candidate.get("max_resampled_rows"),
        "sampling_strategy": deepcopy(candidate.get("sampling_strategy_dict")),
    }


def _format_resampling_details(run):
    details = _selected_resampling_details(run)
    parts = [
        f"candidate={details['candidate_id']}",
        f"method={details['method']}",
    ]
    for key in (
        "alpha",
        "under_fraction",
        "target_class_policy",
        "max_expansion_ratio",
        "max_resampled_rows",
    ):
        value = details[key]
        if value is not None:
            parts.append(f"{key}={value}")
    if details["method_params"]:
        parts.append(
            "method_params="
            + json.dumps(details["method_params"], ensure_ascii=False, sort_keys=True)
        )
    if details["sampling_strategy"] is not None:
        parts.append(
            "sampling_strategy="
            + json.dumps(
                details["sampling_strategy"], ensure_ascii=False, sort_keys=True
            )
        )
    return " | ".join(parts)

def _reduction_k_counts(reduced_frame):
    final_k = int(reduced_frame.shape[1])
    audit = getattr(reduced_frame, "attrs", {}).get("feature_selection_audit")
    if not isinstance(audit, dict) or "unclamped_k" not in audit:
        return final_k, final_k
    try:
        unclamped_k = int(audit["unclamped_k"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Reducer audit contains an invalid unclamped_k.") from exc
    if unclamped_k < 0:
        raise RuntimeError("Reducer audit contains a negative unclamped_k.")
    return final_k, unclamped_k


def _locked_feature_frames(
    X_train,
    X_eval,
    selected_features,
    feature_selection_audit,
):
    """Apply one TRAIN-derived feature list without recomputing RF/MI."""
    selected = list(selected_features)
    if not selected or len(selected) != len(set(selected)):
        raise RuntimeError("Locked feature selection must contain unique columns.")
    missing_train = [column for column in selected if column not in X_train.columns]
    missing_eval = [column for column in selected if column not in X_eval.columns]
    if missing_train or missing_eval:
        raise RuntimeError(
            "Locked feature selection is incompatible with the preprocessed schema: "
            f"missing TRAIN={missing_train}, evaluation={missing_eval}."
        )

    audit = (
        deepcopy(feature_selection_audit)
        if isinstance(feature_selection_audit, dict)
        else {}
    )
    try:
        unclamped_k = int(audit.get("unclamped_k", len(selected)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Locked feature audit contains an invalid unclamped_k.") from exc
    if unclamped_k != len(selected):
        raise RuntimeError(
            "Without K clamping, locked feature width must equal unclamped_k."
        )
    audit.update({
        "unclamped_k": len(selected),
        "final_k": len(selected),
        "features_kept": len(selected),
        "feature_selection_source_candidate": "none",
        "feature_selection_locked": True,
    })

    train_locked = X_train.loc[:, selected].copy()
    eval_locked = X_eval.loc[:, selected].copy()
    train_locked.attrs = deepcopy(getattr(X_train, "attrs", {}))
    eval_locked.attrs = deepcopy(getattr(X_eval, "attrs", {}))
    train_locked.attrs["feature_selection_audit"] = deepcopy(audit)
    eval_locked.attrs["feature_selection_audit"] = deepcopy(audit)
    return train_locked, eval_locked

def _dataframe_sha256(frame):
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(
        frame.reset_index(drop=True), index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _split_class_counts(frame, target_column):
    counts = frame[target_column].value_counts(dropna=False)
    return {str(label): int(count) for label, count in counts.items()}


def _fixed_split_manifest_path(config):
    safe_target = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(config.target_column)
    )
    filename = (
        f"{config.dataset_name}_{config.classification_type}_{safe_target}_"
        f"val20_seed{_RANDOM_SEED}.json"
    )
    return _SPLIT_DIR / filename


def _validate_split_positions(inner_positions, validation_positions, source_rows):
    combined = list(inner_positions) + list(validation_positions)
    if (
        not inner_positions
        or not validation_positions
        or len(combined) != source_rows
        or len(set(combined)) != source_rows
        or sorted(combined) != list(range(source_rows))
    ):
        raise RuntimeError(
            "Fixed validation split JSON does not contain one exact partition "
            "of all source row positions."
        )


def _load_or_create_fixed_validation_split(config, split_source, splitter):
    source = split_source.reset_index(drop=True)
    source_rows = len(source)
    source_sha256 = _dataframe_sha256(source)
    manifest_path = _fixed_split_manifest_path(config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "schema_version": 1,
            "dataset_name": config.dataset_name,
            "classification_type": config.classification_type,
            "target_column": config.target_column,
            "validation_fraction": 0.20,
            "random_state": _RANDOM_SEED,
            "source_rows": source_rows,
            "source_dataframe_sha256": source_sha256,
        }
        mismatches = {
            key: {"expected": value, "received": payload.get(key)}
            for key, value in expected_metadata.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Fixed validation split JSON does not match the current dataset/config: "
                f"{mismatches}."
            )
        inner_positions = [int(value) for value in payload["inner_train_positions"]]
        validation_positions = [
            int(value) for value in payload["validation_positions"]
        ]
        status = "reused"
    else:
        marker = "__autodp_source_row_position__"
        while marker in source.columns:
            marker = "_" + marker
        marked = source.copy()
        marked[marker] = np.arange(source_rows, dtype=int)
        inner_marked, validation_marked = splitter(
            marked,
            config.target_column,
            validation_fraction=0.20,
            random_state=_RANDOM_SEED,
        )
        inner_positions = inner_marked.pop(marker).astype(int).tolist()
        validation_positions = validation_marked.pop(marker).astype(int).tolist()
        payload = {
            "schema_version": 1,
            "dataset_name": config.dataset_name,
            "classification_type": config.classification_type,
            "target_column": config.target_column,
            "validation_fraction": 0.20,
            "random_state": _RANDOM_SEED,
            "source_rows": source_rows,
            "source_dataframe_sha256": source_sha256,
            "inner_train_rows": len(inner_positions),
            "validation_rows": len(validation_positions),
            "inner_train_positions": inner_positions,
            "validation_positions": validation_positions,
            "inner_train_class_counts": _split_class_counts(
                inner_marked, config.target_column
            ),
            "validation_class_counts": _split_class_counts(
                validation_marked, config.target_column
            ),
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        status = "created"

    _validate_split_positions(inner_positions, validation_positions, source_rows)
    inner_train = source.iloc[inner_positions].reset_index(drop=True)
    validation = source.iloc[validation_positions].reset_index(drop=True)
    if set(inner_train[config.target_column].astype(str).unique()) != set(
        source[config.target_column].astype(str).unique()
    ):
        raise RuntimeError("Fixed validation split removed a class from inner TRAIN.")

    print(
        f"[Fixed validation split] {status}: {manifest_path}\n"
        f"  inner TRAIN rows={len(inner_train)} | validation rows={len(validation)} "
        f"| source sha256={source_sha256}"
    )
    return inner_train, validation, str(manifest_path), payload


def _validate_artifact_grid(candidates):
    candidate_ids = list(candidates)
    if candidate_ids == ["none"]:
        return "None"
    if candidate_ids != ["none", "mild", "moderate"]:
        raise RuntimeError(
            "Artifact grid must be exactly ['none'] or ['none', 'mild', 'moderate']."
        )
    mild_family = candidates["mild"].get("imbalance_handling")
    moderate_family = candidates["moderate"].get("imbalance_handling")
    if mild_family == "None" or mild_family != moderate_family:
        raise RuntimeError("Artifact mild and moderate candidates must use one family.")
    return str(mild_family)


def _evaluate_validation_artifact(
    preprocessor,
    config,
    inner_train,
    validation,
    candidates,
):
    """Select features once on candidate none, then select resampling by LightGBM."""
    from experiments.pipeline import _preprocess, _rank_by_f1, _reduce_features

    if "none" not in candidates:
        raise RuntimeError("Feature selection requires the canonical candidate 'none'.")

    controller_name = "LightGBM"
    classifier_names = list(BaseModel.classifier_factories())
    preprocess_seconds = {}
    reduction_seconds = {candidate_id: 0.0 for candidate_id in candidates}
    classifier_seconds = {}
    records = []

    def preprocess_candidate(candidate_id):
        started = time.perf_counter()
        cleaned = _clean_outputs(_preprocess(
            preprocessor,
            config,
            inner_train.copy(),
            validation.copy(),
            resampling_candidate_id=candidate_id,
        ))
        preprocess_seconds[candidate_id] = float(time.perf_counter() - started)
        return cleaned

    X_none, y_none, X_val_none, y_val_none = preprocess_candidate("none")
    reduction_started = time.perf_counter()
    X_none_reduced_raw, X_val_none_reduced_raw = _reduce_features(
        preprocessor, X_none, X_val_none, y_none, config, None
    )
    locked_reduction_seconds = float(time.perf_counter() - reduction_started)
    reduction_seconds["none"] = locked_reduction_seconds
    selected_features = list(X_none_reduced_raw.columns)
    feature_selection_audit = deepcopy(
        getattr(X_none_reduced_raw, "attrs", {}).get("feature_selection_audit")
    )
    X_none_reduced, X_val_none_reduced = _locked_feature_frames(
        X_none_reduced_raw,
        X_val_none_reduced_raw,
        selected_features,
        feature_selection_audit,
    )
    adaptive_k, unclamped_k = _reduction_k_counts(X_none_reduced)

    for candidate_id, candidate in candidates.items():
        if candidate_id == "none":
            X_fit, y_fit, X_val, y_val = X_none, y_none, X_val_none, y_val_none
            X_fit_reduced, X_val_reduced = X_none_reduced, X_val_none_reduced
        else:
            X_fit, y_fit, X_val, y_val = preprocess_candidate(candidate_id)
            X_fit_reduced, X_val_reduced = _locked_feature_frames(
                X_fit,
                X_val,
                selected_features,
                feature_selection_audit,
            )

        classifier_started = time.perf_counter()
        candidate_metrics = BaseModel.evaluate_classifier(
            controller_name,
            X_fit_reduced,
            y_fit,
            X_val_reduced,
            y_val,
        )
        classifier_seconds[candidate_id] = float(
            time.perf_counter() - classifier_started
        )
        records.append({
            "candidate_id": candidate_id,
            "method": str(candidate.get("imbalance_handling", "None")),
            "controller_model": controller_name,
            "adaptive_k": adaptive_k,
            "unclamped_k": unclamped_k,
            "feature_selection_source_candidate": "none",
            "expansion_ratio": float(len(y_fit) / max(1, len(inner_train))),
            "runtime_seconds": float(
                preprocess_seconds[candidate_id]
                + locked_reduction_seconds
                + classifier_seconds[candidate_id]
            ),
            **candidate_metrics,
        })

    ranked, ranking_audit = _rank_by_f1(records)
    winner = ranked[0]
    winner_id = str(winner["candidate_id"])
    selected_candidates = {name: winner_id for name in classifier_names}
    shared_preprocess = float(sum(preprocess_seconds.values()))
    shared_reduction = locked_reduction_seconds
    controller_compute = float(sum(classifier_seconds.values()))
    classifier_validation_timing = {
        name: {
            "validation_preprocess_seconds": shared_preprocess,
            "validation_reduction_seconds": shared_reduction,
            "validation_classifier_seconds": (
                controller_compute if name == controller_name else 0.0
            ),
            "validation_non_api_seconds": float(
                shared_preprocess
                + shared_reduction
                + (controller_compute if name == controller_name else 0.0)
            ),
        }
        for name in classifier_names
    }
    unique_non_api_seconds = float(
        shared_preprocess + shared_reduction + controller_compute
    )

    return {
        "selected_candidate": winner_id,
        "selected_candidates": selected_candidates,
        "adaptive_k": adaptive_k,
        "unclamped_k": unclamped_k,
        "selected_features": selected_features,
        "feature_selection_source_candidate": "none",
        "controller_model": controller_name,
        "controller_metrics": {
            controller_name: {
                "winner_id": winner_id,
                "weighted_f1_near_tolerance": ranking_audit["weighted_f1_near_tolerance"],
                "best_weighted_f1": ranking_audit["best_weighted_f1"],
                "weighted_f1_floor": ranking_audit["weighted_f1_floor"],
                "weighted_f1_near_candidate_ids": ranking_audit["weighted_f1_near_candidate_ids"],
                "ranking_metric_after_near_tie": ranking_audit["ranking_metric_after_near_tie"],
                "resampling_intensity_order": ranking_audit["resampling_intensity_order"],
                "candidates": records,
            }
        },
        "classifier_validation_runtime_seconds": {
            name: timing["validation_classifier_seconds"]
            for name, timing in classifier_validation_timing.items()
        },
        "classifier_validation_timing": classifier_validation_timing,
        "validation_operation_timing": {
            "preprocess_by_candidate": preprocess_seconds,
            "reduction_by_candidate": reduction_seconds,
            "feature_selection_source_candidate": "none",
            "controller_classifier_by_candidate": classifier_seconds,
            "unique_non_api_seconds": unique_non_api_seconds,
        },
        "timing_unique_non_api_seconds": unique_non_api_seconds,
    }



def _load_artifact_for_test(config, filename):
    from experiments.pipeline import build_preprocessor

    reuse_config = replace(
        config,
        use_llm=False,
        generated_file=filename,
        generated_files=[],
        n_runs=1,
    )
    return build_preprocessor(reuse_config), reuse_config


def _evaluate_test_artifact(run, config, inner_train, test_df, locked_k=None):
    """Apply the validation-locked feature list to TEST without a full-TRAIN refit."""
    from experiments.pipeline import _preprocess

    test_phase_started = time.perf_counter()
    preprocessor, reuse_config = _load_artifact_for_test(
        config, run["artifact_file"]
    )
    candidate_id = str(
        run.get("selected_candidate")
        or run.get("selected_candidates", {}).get("LightGBM")
    )
    selected_features = run.get("selected_features")
    if not isinstance(selected_features, list) or not selected_features:
        raise RuntimeError(
            "Validation run has no locked selected_features for TEST evaluation."
        )
    expected_k = int(run.get("adaptive_k", len(selected_features)))
    expected_unclamped_k = int(run.get("unclamped_k", expected_k))
    if expected_k != len(selected_features) or expected_unclamped_k != expected_k:
        raise RuntimeError(
            "Locked validation K must equal selected feature count without clamping."
        )
    if locked_k is not None and int(locked_k) != expected_k:
        raise RuntimeError("Requested locked_k conflicts with validation-selected K.")

    preprocess_started = time.perf_counter()
    X_fit, y_fit, X_test, y_test = _clean_outputs(_preprocess(
        preprocessor,
        reuse_config,
        inner_train.copy(),
        test_df.copy(),
        resampling_candidate_id=candidate_id,
    ))
    preprocess_seconds = float(time.perf_counter() - preprocess_started)

    reduction_started = time.perf_counter()
    X_fit_reduced, X_test_reduced = _locked_feature_frames(
        X_fit,
        X_test,
        selected_features,
        {
            "unclamped_k": expected_unclamped_k,
            "feature_selection_source_candidate": "none",
        },
    )
    reduction_seconds = float(time.perf_counter() - reduction_started)
    adaptive_k, unclamped_k = _reduction_k_counts(X_fit_reduced)

    results = {}
    failures = {}
    classifier_wall_seconds = {}
    for classifier_name in BaseModel.classifier_factories():
        try:
            classifier_started = time.perf_counter()
            evaluated = BaseModel.evaluate_classifier(
                classifier_name,
                X_fit_reduced,
                y_fit,
                X_test_reduced,
                y_test,
            )
            classifier_wall_seconds[classifier_name] = float(
                time.perf_counter() - classifier_started
            )
            results[classifier_name] = {
                "selected_k": adaptive_k,
                "unclamped_k": unclamped_k,
                "candidate_id": candidate_id,
                **evaluated,
            }
        except Exception as exc:
            failures[classifier_name] = f"{type(exc).__name__}: {exc}"

    classifier_test_timing = {}
    for classifier_name in BaseModel.classifier_factories():
        classifier_seconds = float(classifier_wall_seconds.get(classifier_name, 0.0))
        classifier_test_timing[classifier_name] = {
            "test_preprocess_seconds": preprocess_seconds,
            "test_reduction_seconds": reduction_seconds,
            "test_classifier_seconds": classifier_seconds,
            "test_non_api_seconds": float(
                preprocess_seconds + reduction_seconds + classifier_seconds
            ),
        }

    unique_non_api_seconds = float(
        preprocess_seconds
        + reduction_seconds
        + sum(classifier_wall_seconds.values())
    )
    return {
        "run_index": run["run_index"],
        "artifact_file": run["artifact_file"],
        "seed": run["seed"],
        "unclamped_k": unclamped_k,
        "selected_candidate": candidate_id,
        "adaptive_k": adaptive_k,
        "selected_features": list(selected_features),
        "feature_selection_source_candidate": "none",
        "training_scope": "fixed_inner_train_80_percent",
        "fit_source_rows": int(len(inner_train)),
        "metrics": results,
        "failures": failures,
        "classifier_test_timing": classifier_test_timing,
        "test_operation_timing": {
            "preprocess_by_candidate": {candidate_id: preprocess_seconds},
            "reduction_by_candidate": {candidate_id: reduction_seconds},
            "feature_selection_source_candidate": "none",
            "unique_non_api_seconds": unique_non_api_seconds,
        },
        "timing_unique_non_api_seconds": unique_non_api_seconds,
        "test_phase_wall_seconds": float(
            time.perf_counter() - test_phase_started
        ),
    }


_TIMING_COMPONENT_KEYS = (
    "validation_preprocess_seconds",
    "validation_reduction_seconds",
    "validation_classifier_seconds",
    "test_preprocess_seconds",
    "test_reduction_seconds",
    "test_classifier_seconds",
)


def _artifact_classifier_timing_records(validation_runs, test_runs):
    """Return one complete generation/VAL/TEST timing sample per artifact."""
    test_by_run = {run.get("run_index"): run for run in test_runs}
    records = {name: [] for name in BaseModel.classifier_factories()}

    for validation_run in validation_runs:
        run_index = validation_run.get("run_index")
        test_run = test_by_run.get(run_index)
        if test_run is None:
            continue

        api_seconds = max(
            0.0,
            float(validation_run.get("llm_usage", {}).get("api_latency_seconds", 0.0)),
        )
        validation_unique = float(
            validation_run.get("timing_unique_non_api_seconds", 0.0)
        )
        test_unique = float(test_run.get("timing_unique_non_api_seconds", 0.0))
        validation_wall = float(
            validation_run.get(
                "validation_phase_wall_seconds",
                api_seconds + validation_unique,
            )
        )
        test_wall = float(
            test_run.get("test_phase_wall_seconds", test_unique)
        )
        common_non_api = max(
            0.0,
            validation_wall + test_wall
            - api_seconds - validation_unique - test_unique,
        )

        for classifier_name in records:
            if classifier_name not in test_run.get("metrics", {}):
                continue
            validation_timing = validation_run.get(
                "classifier_validation_timing", {}
            ).get(classifier_name, {})
            test_timing = test_run.get("classifier_test_timing", {}).get(
                classifier_name, {}
            )
            record = {
                "run_index": run_index,
                "artifact_file": validation_run.get("artifact_file"),
                "seed": validation_run.get("seed"),
                "api_latency_seconds": api_seconds,
                "common_non_api_seconds": common_non_api,
            }
            for key in _TIMING_COMPONENT_KEYS[:3]:
                record[key] = float(validation_timing.get(key, 0.0))
            for key in _TIMING_COMPONENT_KEYS[3:]:
                record[key] = float(test_timing.get(key, 0.0))
            record["classifier_compute_seconds"] = float(
                record["validation_classifier_seconds"]
                + record["test_classifier_seconds"]
            )
            attributed = sum(record[key] for key in _TIMING_COMPONENT_KEYS)
            record["per_model_non_api_seconds"] = float(
                common_non_api + attributed
            )
            record["end_to_end_seconds"] = float(
                api_seconds + record["per_model_non_api_seconds"]
            )
            records[classifier_name].append(record)

    return records


def _mean_std(records, key):
    values = np.asarray([float(record[key]) for record in records], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _format_timing_mean_std(result, key):
    mean_key = f"{key}_mean"
    std_key = f"{key}_std"
    return f"{result[mean_key]:.2f} ± {result[std_key]:.2f}"


def _aggregate_test(
    test_runs,
    locked_k=None,
    validation_runs=None,
    api_to_test_wall_seconds=None,
    api_latency_seconds=0.0,
):
    validation_runs = validation_runs or []
    classifier_names = list(BaseModel.classifier_factories())
    artifact_timing_records = _artifact_classifier_timing_records(
        validation_runs, test_runs
    )
    api_latency_seconds = max(0.0, float(api_latency_seconds))
    unique_tracked_non_api_seconds = float(
        sum(
            float(run.get("timing_unique_non_api_seconds", 0.0))
            for run in validation_runs
        )
        + sum(
            float(run.get("timing_unique_non_api_seconds", 0.0))
            for run in test_runs
        )
    )
    if api_to_test_wall_seconds is None:
        common_non_api_seconds = float("nan")
    else:
        common_non_api_seconds = max(
            0.0,
            float(api_to_test_wall_seconds)
            - api_latency_seconds
            - unique_tracked_non_api_seconds,
        )

    classifier_timing = {}
    for classifier_name in classifier_names:
        timing = {
            "validation_preprocess_seconds": 0.0,
            "validation_reduction_seconds": 0.0,
            "validation_classifier_seconds": 0.0,
            "test_preprocess_seconds": 0.0,
            "test_reduction_seconds": 0.0,
            "test_classifier_seconds": 0.0,
        }
        for run in validation_runs:
            run_timing = run.get("classifier_validation_timing", {}).get(
                classifier_name, {}
            )
            for key in (
                "validation_preprocess_seconds",
                "validation_reduction_seconds",
                "validation_classifier_seconds",
            ):
                timing[key] += float(run_timing.get(key, 0.0))
        for run in test_runs:
            run_timing = run.get("classifier_test_timing", {}).get(
                classifier_name, {}
            )
            for key in (
                "test_preprocess_seconds",
                "test_reduction_seconds",
                "test_classifier_seconds",
            ):
                timing[key] += float(run_timing.get(key, 0.0))
        timing["classifier_compute_seconds"] = float(
            timing["validation_classifier_seconds"]
            + timing["test_classifier_seconds"]
        )
        attributed_operations = sum(
            timing[key]
            for key in (
                "validation_preprocess_seconds",
                "validation_reduction_seconds",
                "validation_classifier_seconds",
                "test_preprocess_seconds",
                "test_reduction_seconds",
                "test_classifier_seconds",
            )
        )
        timing["per_model_non_api_seconds"] = float(
            common_non_api_seconds + attributed_operations
            if np.isfinite(common_non_api_seconds)
            else float("nan")
        )
        timing["end_to_end_seconds"] = float(
            api_latency_seconds + timing["per_model_non_api_seconds"]
            if np.isfinite(timing["per_model_non_api_seconds"])
            else float("nan")
        )
        classifier_timing[classifier_name] = timing

    rows = []
    aggregate = {}
    for classifier_name in classifier_names:
        records = [
            run["metrics"][classifier_name]
            for run in test_runs
            if classifier_name in run["metrics"]
        ]
        timing = classifier_timing[classifier_name]
        per_artifact_timing = artifact_timing_records[classifier_name]
        timing_statistics = {}
        for key in (
            "api_latency_seconds",
            "common_non_api_seconds",
            *_TIMING_COMPONENT_KEYS,
            "classifier_compute_seconds",
            "per_model_non_api_seconds",
            "end_to_end_seconds",
        ):
            mean, std = _mean_std(per_artifact_timing, key)
            timing_statistics[f"{key}_mean"] = mean
            timing_statistics[f"{key}_std"] = std
        common_timing = {
            "api_latency_seconds": api_latency_seconds,
            "common_non_api_seconds": float(common_non_api_seconds),
            "joint_unique_tracked_non_api_seconds": unique_tracked_non_api_seconds,
            "n_timed_artifacts": len(per_artifact_timing),
            "per_artifact_timing": per_artifact_timing,
            **timing_statistics,
            **timing,
        }
        selected_k_values = sorted({
            int(record["selected_k"] if "selected_k" in record else locked_k[classifier_name])
            for record in records
        }) if records else []
        selected_k = selected_k_values[0] if len(selected_k_values) == 1 else None
        unclamped_k_values = sorted({
            int(
                record["unclamped_k"]
                if "unclamped_k" in record
                else record["selected_k"] if "selected_k" in record else locked_k[classifier_name]
            )
            for record in records
        }) if records else []
        unclamped_k = unclamped_k_values[0] if len(unclamped_k_values) == 1 else None
        if records:
            accuracy = np.asarray(
                [float(record["accuracy"]) for record in records], dtype=float
            )
            weighted_f1 = np.asarray(
                [float(record["weighted_f1"]) for record in records], dtype=float
            )
            weighted_precision = np.asarray(
                [float(record["weighted_precision"]) for record in records],
                dtype=float,
            )
            weighted_recall = np.asarray(
                [float(record["weighted_recall"]) for record in records],
                dtype=float,
            )
            result = {
                "n_successful_test_runs": int(len(records)),
                "selected_k": selected_k,
                "selected_k_values": selected_k_values,
                "unclamped_k": unclamped_k,
                "unclamped_k_values": unclamped_k_values,
                "accuracy_mean": float(np.mean(accuracy)),
                "accuracy_std": float(np.std(accuracy, ddof=0)),
                "weighted_f1_mean": float(np.mean(weighted_f1)),
                "weighted_f1_std": float(np.std(weighted_f1, ddof=0)),
                "weighted_precision_mean": float(np.mean(weighted_precision)),
                "weighted_precision_std": float(
                    np.std(weighted_precision, ddof=0)
                ),
                "weighted_recall_mean": float(np.mean(weighted_recall)),
                "weighted_recall_std": float(np.std(weighted_recall, ddof=0)),
                **common_timing,
            }
        else:
            result = {
                "n_successful_test_runs": 0,
                "selected_k": selected_k,
                "selected_k_values": selected_k_values,
                "unclamped_k": unclamped_k,
                "unclamped_k_values": unclamped_k_values,
                "accuracy_mean": float("nan"),
                "accuracy_std": float("nan"),
                "weighted_f1_mean": float("nan"),
                "weighted_f1_std": float("nan"),
                "weighted_precision_mean": float("nan"),
                "weighted_precision_std": float("nan"),
                "weighted_recall_mean": float("nan"),
                "weighted_recall_std": float("nan"),
                **common_timing,
            }
        aggregate[classifier_name] = result
        rows.append({
            "Classifier": classifier_name,
            "Selected K": (
                str(result["selected_k"])
                if result["selected_k"] is not None
                else ",".join(str(value) for value in result["selected_k_values"])
            ),
            "Unclamped K": (
                str(result["unclamped_k"])
                if result["unclamped_k"] is not None
                else ",".join(str(value) for value in result["unclamped_k_values"])
            ),
            "Accuracy mean ± std": f"{result['accuracy_mean']:.6f} ± {result['accuracy_std']:.6f}",
            "Weighted Precision mean ± std": f"{result['weighted_precision_mean']:.6f} ± {result['weighted_precision_std']:.6f}",
            "Weighted Recall mean ± std": f"{result['weighted_recall_mean']:.6f} ± {result['weighted_recall_std']:.6f}",
            "Weighted F1 mean ± std": f"{result['weighted_f1_mean']:.6f} ± {result['weighted_f1_std']:.6f}",
            "Timed artifacts": result["n_timed_artifacts"],
            "API/artifact mean ± std (s)": _format_timing_mean_std(result, "api_latency_seconds"),
            "Common non-API/artifact mean ± std (s)": _format_timing_mean_std(result, "common_non_api_seconds"),
            "VAL preprocess/artifact mean ± std (s)": _format_timing_mean_std(result, "validation_preprocess_seconds"),
            "VAL reduction/artifact mean ± std (s)": _format_timing_mean_std(result, "validation_reduction_seconds"),
            "VAL classifier/artifact mean ± std (s)": _format_timing_mean_std(result, "validation_classifier_seconds"),
            "TEST preprocess/artifact mean ± std (s)": _format_timing_mean_std(result, "test_preprocess_seconds"),
            "TEST reduction/artifact mean ± std (s)": _format_timing_mean_std(result, "test_reduction_seconds"),
            "TEST classifier/artifact mean ± std (s)": _format_timing_mean_std(result, "test_classifier_seconds"),
            "Classifier compute/artifact mean ± std (s)": _format_timing_mean_std(result, "classifier_compute_seconds"),
            "Non-API/artifact mean ± std (s)": _format_timing_mean_std(result, "per_model_non_api_seconds"),
            "Total/artifact mean ± std (s)": _format_timing_mean_std(result, "end_to_end_seconds"),
        })
    return aggregate, pd.DataFrame(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save_suite_manifest(config, suite_result, report_df):
    _RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = (
        f"{config.dataset_name}_{config.model_id.replace('/', '_')}_"
        f"{config.classification_type}_validation_suite_{timestamp}"
    )
    manifest_path = _RESULTS_DIR / f"{stem}.json"
    report_path = _RESULTS_DIR / f"{stem}.txt"
    manifest_path.write_text(
        json.dumps(_json_safe(suite_result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"Dataset: {config.dataset_name}",
        f"Fixed validation split JSON: {suite_result['validation_split_manifest_path']}",
        "Final fit scope: fixed inner TRAIN 80%",
        "Full-TRAIN refit after validation: no",
        f"Artifacts requested: {config.n_runs}",
        f"Artifacts validation-valid: {len(suite_result['validation_runs'])}",
        "Population standard deviation: ddof=0",
        "Classification averaging: precision/recall/F1 = weighted",
        "Selection: weighted-F1 difference < 0.003 -> lower intensity (none < mild < moderate)",
        "Adaptive K: selected once on candidate none; RF+MI score >= 0.55; Pearson redundancy >= 0.90; no threshold relaxation; no K clamp",
        "Resampling grid controller: standard LightGBM only",
        f"End-to-end timing origin: {suite_result['end_to_end_timing']['origin']}",
        "Timing unit: one completed artifact = generation/load + VAL + its own TEST",
        "Timing aggregation: mean ± population std across completed TEST artifacts",
        "Per artifact: common_non_API = artifact_wall - API_latency - unique_tracked_operations",
        "Per artifact: model_non_API = common_non_API + operations consumed by classifier",
        "Per artifact: total = API_latency + model_non_API",
        "",
        "Locked resampling by artifact:",
        *[
            f"  {run.get('artifact_id', f'artifact_{index:02d}')}: "
            f"{_format_resampling_details(run)}"
            for index, run in enumerate(suite_result["validation_runs"], start=1)
        ],
        "",
        "Final K from locked inner TRAIN by artifact:",
        *[
            f"  {artifact_id}: K={final_k} | unclamped K="
            f"{suite_result['final_unclamped_k_by_artifact'][artifact_id]}"
            for artifact_id, final_k in suite_result["final_k_by_artifact"].items()
        ],
        "",
        report_df.to_string(index=False),
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(manifest_path), str(report_path)


def run_validation_suite(config: ExperimentConfig):
    """Generate/reuse artifacts, select resampling with LightGBM, then open TEST once."""
    from experiments.pipeline import (
        _generated_resampling_contract,
        _normalize_target_column,
        _prepare_train_for_selection_split,
        _rare_safe_train_validation_split,
        build_preprocessor,
    )

    config = _normalize_target_column(config)

    started = time.perf_counter()
    end_to_end_started = None
    end_to_end_origin = None
    print("Loading full TRAIN metadata (official TEST remains unopened)...")
    train_df = pd.read_csv(config.train_path)
    split_source = _prepare_train_for_selection_split(config, train_df)
    inner_train, validation, validation_split_manifest_path, validation_split = (
        _load_or_create_fixed_validation_split(
            config, split_source, _rare_safe_train_validation_split
        )
    )
    if (
        config.dataset_name == "NSL_KDD"
        and config.classification_type == "multiclass"
        and inner_train[config.target_column].nunique() != 5
    ):
        raise RuntimeError("NSL-KDD rare-safe split did not preserve all five classes.")

    validation_runs = []
    failed_runs = []
    aggregate_usage = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_latency_seconds": 0.0,
        "cost_usd": 0.0,
    }

    for run_index in range(config.n_runs):
        seed = _RANDOM_SEED + run_index
        usage = {}
        artifact_phase_started = time.perf_counter()
        try:
            run_config = config
            if not config.use_llm:
                filenames = (
                    config.generated_files
                    if config.n_runs > 1
                    else [config.generated_file]
                )
                run_config = replace(
                    config,
                    generated_file=filenames[run_index],
                    generated_files=[],
                    n_runs=1,
                )
            print(
                f"\n[Artifact {run_index + 1}/{config.n_runs}] "
                f"seed={seed}"
            )
            if end_to_end_started is None:
                end_to_end_started = time.perf_counter()
                end_to_end_origin = (
                    "first_llm_generation_api_phase_start"
                    if config.use_llm
                    else "first_reused_artifact_load_start"
                )
            preprocessor = build_preprocessor(
                run_config,
                train_df=inner_train,
                llm_usage=usage,
                test_df=None,
                generation_seed=seed,
            )
            candidates, selection = _generated_resampling_contract(preprocessor)
            if candidates is None:
                raise RuntimeError("Generated artifact has no resampling contract.")
            family = _validate_artifact_grid(candidates)
            validation_result = _evaluate_validation_artifact(
                preprocessor,
                run_config,
                inner_train,
                validation,
                candidates,
            )
            run_record = {
                "run_index": run_index + 1,
                "artifact_id": f"artifact_{run_index + 1:02d}",
                "artifact_file": _artifact_filename(preprocessor),
                "seed": seed,
                "resampling_family": family,
                "resampling_candidates": candidates,
                "resampling_selection_contract": selection,
                "llm_usage": usage,
                "validation_phase_wall_seconds": float(
                    time.perf_counter() - artifact_phase_started
                ),
                **validation_result,
            }
            validation_runs.append(run_record)
            for key in aggregate_usage:
                aggregate_usage[key] += usage.get(key, 0)
        except Exception as exc:
            for key in aggregate_usage:
                aggregate_usage[key] += usage.get(key, 0)
            failed_runs.append({
                "run_index": run_index + 1,
                "seed": seed,
                "phase": "generation_or_validation",
                "error": f"{type(exc).__name__}: {exc}",
                "llm_usage": usage,
                "validation_phase_wall_seconds": float(
                    time.perf_counter() - artifact_phase_started
                ),
            })
            print(
                f"[Artifact {run_index + 1}] rejected: "
                f"{type(exc).__name__}: {exc}"
            )

    if not validation_runs:
        raise RuntimeError(
            "No artifact completed the full validation grid; official TEST was not read."
        )

    validation_adaptive_k_by_artifact = {
        str(run["artifact_id"]): int(run["adaptive_k"])
        for run in validation_runs
    }
    validation_unclamped_k_by_artifact = {
        str(run["artifact_id"]): int(run["unclamped_k"])
        for run in validation_runs
    }
    print("\nResampling winners locked by standard LightGBM; official TEST remains unopened:")
    for run in validation_runs:
        print(
            f"  {run['artifact_id']}: candidate={run['selected_candidate']} "
            f"| validation adaptive K={run['adaptive_k']} | unclamped K={run['unclamped_k']}"
        )
        print(f"    resampling: {_format_resampling_details(run)}")

    print("\nLoading official TEST; fitting remains locked to inner TRAIN 80%...")
    test_df = pd.read_csv(config.test_path)
    test_runs = []
    for run in validation_runs:
        test_record = _evaluate_test_artifact(
            run,
            config,
            inner_train,
            test_df,
        )
        if (
            int(test_record.get("adaptive_k", -1)) != int(run["adaptive_k"])
            or int(test_record.get("unclamped_k", -1)) != int(run["unclamped_k"])
            or list(test_record.get("selected_features", []))
            != list(run.get("selected_features", []))
            or str(test_record.get("selected_candidate"))
            != str(run["selected_candidate"])
        ):
            raise RuntimeError(
                "Official TEST evaluation changed the validation-locked feature "
                "selection or resampling candidate."
            )
        test_runs.append(test_record)
        for classifier_name, error in test_record["failures"].items():
            failed_runs.append({
                "run_index": run["run_index"],
                "seed": run["seed"],
                "phase": "test",
                "classifier": classifier_name,
                "error": error,
            })

    final_k_by_artifact = {
        str(validation_run["artifact_id"]): int(test_run["adaptive_k"])
        for validation_run, test_run in zip(validation_runs, test_runs)
    }
    final_unclamped_k_by_artifact = {
        str(validation_run["artifact_id"]): int(test_run["unclamped_k"])
        for validation_run, test_run in zip(validation_runs, test_runs)
    }
    print("\nFinal K from locked inner TRAIN (no full-TRAIN refit):")
    for validation_run, test_run in zip(validation_runs, test_runs):
        print(
            f"  {validation_run['artifact_id']} "
            f"({validation_run['artifact_file']}): "
            f"final K={test_run['adaptive_k']} "
            f"| unclamped K={test_run['unclamped_k']} "
            f"| candidate={test_run['selected_candidate']}"
        )
        print(f"    resampling: {_format_resampling_details(validation_run)}")

    api_to_test_wall_seconds = float(time.perf_counter() - end_to_end_started)
    aggregate_metrics, report_df = _aggregate_test(
        test_runs,
        validation_runs=validation_runs,
        api_to_test_wall_seconds=api_to_test_wall_seconds,
        api_latency_seconds=aggregate_usage.get("api_latency_seconds", 0.0),
    )
    suite_result = {
        "validation_runs": validation_runs,
        "validation_split_manifest_path": validation_split_manifest_path,
        "validation_split": validation_split,
        "training_protocol": {
            "selection_fit_scope": "fixed_inner_train_80_percent",
            "validation_scope": "fixed_train_holdout_20_percent",
            "full_train_refit": False,
            "official_test_evaluations": 1,
        },
        "validation_adaptive_k_by_artifact": validation_adaptive_k_by_artifact,
        "validation_unclamped_k_by_artifact": validation_unclamped_k_by_artifact,
        # Compatibility key reflects final-TEST adaptive K rather than the
        # earlier TRAIN-derived validation K.
        "adaptive_k_by_artifact": final_k_by_artifact,
        "final_k_by_artifact": final_k_by_artifact,
        "final_unclamped_k_by_artifact": final_unclamped_k_by_artifact,
        "feature_selection_contract": {
            "mode": "adaptive_rf_mi_threshold",
            "rf_mi_threshold": 0.55,
            "completion_corr_threshold": 0.90,
            "semantic_rf_hyperparameters": {"n_estimators": 50, "max_depth": 8, "class_weight": "balanced_subsample", "random_state": 42, "n_jobs": None},
            "all_feature_rf_hyperparameters": {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced_subsample", "random_state": 42, "n_jobs": None},
            "rf_weight": 0.60,
            "mi_weight": 0.40,
            "source_candidate": "none",
            "k_clamping": False,
            "locked_across_resampling_candidates": True,
            "locked_for_official_test": True,
        },
        "resampling_controller_contract": {
            "model": "LightGBM",
            "configuration": "LGBMClassifier(random_state=42, verbosity=-1)",
            "weighted_f1_near_tolerance": _WEIGHTED_F1_NEAR_TOLERANCE,
            "tie_breaker": "lower_resampling_intensity",
        },
        "locked_resampling_by_artifact": {
            str(run["artifact_id"]): _selected_resampling_details(run)
            for run in validation_runs
        },
        "test_runs": test_runs,
        "aggregate_metrics": aggregate_metrics,
        "failed_runs": failed_runs,
        "llm_usage": aggregate_usage,
        "wall_clock_seconds": float(time.perf_counter() - started),
        "api_to_test_wall_seconds": api_to_test_wall_seconds,
        "end_to_end_timing": {
            "origin": (
                "each_artifact_generation_api_phase_start"
                if config.use_llm
                else "each_reused_artifact_load_start"
            ),
            "description": (
                "Per-artifact operation-consumer accounting. Each timing sample "
                "combines that artifact generation, validation, and its own TEST; "
                "cached costs are assigned only to classifiers that consume them."
                if config.use_llm
                else
                "Per-artifact operation-consumer accounting for reused artifacts. "
                "Each sample combines artifact load, validation, and its own TEST; "
                "no API call occurred."
            ),
        },
    }
    manifest_path, report_path = _save_suite_manifest(
        config, suite_result, report_df
    )
    suite_result["manifest_path"] = manifest_path
    suite_result["report_path"] = report_path
    suite_result["report"] = report_df

    print("\nMain TEST report:")
    print(report_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\n[Audit manifest] {manifest_path}")
    print(f"[Main report] {report_path}")
    return suite_result
