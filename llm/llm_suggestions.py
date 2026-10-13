"""
LLM suggestion utilities
"""
import os
import re
import ast
import json
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from scipy.stats import chi2_contingency, f_oneway
from .api_caller import APICaller
from .prompt_builders import (
    build_base_context,
    build_dataset_context,
    build_stage2_handoff,
    build_stage1_system_prompt,
    build_stage1_user_prompt,
    build_stage2_system_prompt,
    build_stage2_user_prompt,
    build_stage3_system_prompt,
    build_stage3_user_prompt,
)
from .validation import (
    error as validation_error,
    format_feedback,
    parse_strict_json_object,
    parse_strict_python_response,
    validate_preprocessor_response,
    validate_reducer_response,
)

load_dotenv()

_GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated"
_HISTORY_FILE = _GENERATED_DIR / "history.json"
_REJECTED_DIR = Path(__file__).resolve().parents[1] / "rejected_artifacts"

# Skip chi-square based association on very high-cardinality columns to avoid huge
# contingency tables (OOM) — these are typically ID-like columns, not useful categoricals.
_MAX_CARD_FOR_CHI2 = 1000

# --- Prompt context-budget guards ---------------------------------------------------------------
# IDS datasets can be very wide; rendering every stat/pair into the prompt can blow up the context
# window, raise API cost, and dilute the LLM's focus. We cap how many rows of each table are shown.
_MAX_PAIR_ROWS = 60       # cap for the (quadratic) feature-PAIR association tables
_MAX_FEATURE_ROWS = 200   # cap for the (linear) per-FEATURE tables (descriptive stats, target assoc)
_NZV_STD_EPS = 1e-9       # numeric std at/below this ⇒ treated as (near-)zero-variance

_CATEGORICAL_ENCODING_POLICY = {
    "binary": "zero_one",
    "low_cardinality_max": 16,
    "low_cardinality_method": "onehot",
    "onehot_unknown_strategy": "explicit_unknown_bucket",
    "onehot_unknown_token": "__UNKNOWN__",
    "high_cardinality_method": "cross_fitted_multiclass_target",
    "target_encoder_cv": 5,
    "target_encoder_smooth": "auto",
    "target_encoder_shuffle": True,
    "target_encoder_random_state": 42,
    "fit_scope": "train_only",
}

_SCALING_POLICY = {
    "method": "StandardScaler",
    "scale_continuous_numeric": True,
    "scale_target_encoded": True,
    "scale_binary": False,
    "scale_onehot": False,
    "fit_before_resampling": True,
    "fit_scope": "train_only",
}

_LOG_TRANSFORM_POLICY = {
    "method": "train_only_right_skew_rule",
    "minimum_unique_values": 20,
    "minimum_skewness": 2.0,
    "upper_tail_iqr_multiplier": 3.0,
}


def _categorical_encoding_method(cardinality: int) -> str:
    """Return the fixed TRAIN-cardinality-derived categorical encoding method."""
    if cardinality == 2:
        return "zero_one"
    if 3 <= cardinality <= _CATEGORICAL_ENCODING_POLICY["low_cardinality_max"]:
        return "onehot"
    if cardinality > _CATEGORICAL_ENCODING_POLICY["low_cardinality_max"]:
        return "cross_fitted_multiclass_target"
    raise ValueError(
        "Categorical features with fewer than two non-null TRAIN categories must be "
        "removed as verified constant/all-missing features before encoding."
    )


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V (chi-square based) association between two categorical series. Range [0, 1]."""
    try:
        confusion = pd.crosstab(x, y)
        if confusion.shape[0] < 2 or confusion.shape[1] < 2:
            return float('nan')
        chi2 = chi2_contingency(confusion)[0]
        n = confusion.to_numpy().sum()
        phi2 = chi2 / n
        r, k = confusion.shape
        phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        return float(np.sqrt(phi2corr / denom)) if denom > 0 else float('nan')
    except Exception:
        return float('nan')


def _chi2_pvalue(x: pd.Series, y: pd.Series) -> float:
    """Chi-square test of independence p-value between two categorical series."""
    try:
        confusion = pd.crosstab(x, y)
        if confusion.shape[0] < 2 or confusion.shape[1] < 2:
            return float('nan')
        return float(chi2_contingency(confusion)[1])
    except Exception:
        return float('nan')


def _anova_f(num_series: pd.Series, cat_series: pd.Series):
    """One-way ANOVA: F-statistic and p-value of a numeric variable grouped by a categorical variable."""
    try:
        df = pd.DataFrame({'num': num_series, 'cat': cat_series}).dropna()
        groups = [g['num'].to_numpy() for _, g in df.groupby('cat', observed=True) if len(g) > 1]
        if len(groups) < 2:
            return float('nan'), float('nan')
        f, p = f_oneway(*groups)
        return float(f), float(p)
    except Exception:
        return float('nan'), float('nan')

class LLM_Suggestions:
    def __init__(self, model_name):
        self.model_name = model_name
        provider = os.environ.get("LLM_PROVIDER", "openai")
        self.api_caller = APICaller(
            base_url=(
                os.environ.get("BASE_URL")
                or os.environ.get("ANTHROPIC_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or None
            ),
            api_key=(
                os.environ.get("API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or ""
            ),
            model_name=model_name,
            provider=provider,
        )

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Return the contents of the first ``` fenced python block, or the raw stripped text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Cannot extract code because the LLM response is empty or non-text."
            )
        pattern = r"`" * 3 + r"(?:python)?\s*(.*?)" + r"`" * 3
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def _compose_generated_module(
        stage2_response: str,
        stage3_response: str,
    ) -> tuple[str, str]:
        """Build and syntax-check the complete Stage-2/Stage-3 module in memory."""
        cleaning_code = LLM_Suggestions._extract_code_block(stage2_response)
        reduction_code = LLM_Suggestions._extract_code_block(stage3_response)
        kind_match = re.search(
            r"REDUCTION_KIND\s*(?::\s*\w+)?\s*=\s*[\"']([A-Za-z_]+)[\"']",
            reduction_code,
        )
        reduction_kind = kind_match.group(1).lower() if kind_match else "selection"
        module_parts = [
            cleaning_code,
            "",
            "# === Stage 3: LLM-designed feature reduction "
            "(REDUCTION_KIND, reduce_features) ===",
            reduction_code,
        ]
        if not re.search(
            r"(?m)^\s*REDUCTION_KIND\s*(?::\s*\w+)?\s*=", reduction_code
        ):
            module_parts.extend(["", f'REDUCTION_KIND = "{reduction_kind}"'])
        module_parts.append("")
        module_code = "\n".join(module_parts)
        try:
            ast.parse(module_code)
        except SyntaxError as exc:
            raise RuntimeError(
                "The validated Stage-2 and Stage-3 responses do not form a valid "
                f"combined module: {exc}"
            ) from exc
        return module_code, reduction_kind

    @staticmethod
    def _write_generated_file(code_text: str, model_name: str, classification_type: str) -> tuple:
        """Atomically publish one complete, validated generated module."""
        _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = re.sub(r"[^a-zA-Z0-9_]", "_", model_name)
        # The reduction method is now CHOSEN by the LLM (Stage 3), so the file/cache token is the
        # constant "llm"; the concrete kind is recorded separately as `reduction_kind`.
        filename = f"preprocessing_{timestamp}_{safe_model}_{classification_type}_llm.py"
        target_path = _GENERATED_DIR / filename
        temporary_path = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
        try:
            temporary_path.write_text(
                LLM_Suggestions._extract_code_block(code_text), encoding="utf-8"
            )
            temporary_path.replace(target_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return filename, timestamp

    @staticmethod
    def _record_history(timestamp: str, filename: str, model_name: str, dataset_name: str,
                        classification_type: str, reduction_kind: str) -> None:
        """Append a generation-history entry. `reduction_kind` stores the LLM's actual choice (selection/extraction/hybrid) for info."""
        if _HISTORY_FILE.exists():
            history = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
        else:
            history = []
        history.append({
            "id": timestamp,
            "file": filename,
            "model": model_name,
            "dataset_name": dataset_name,
            "classification_type": classification_type,
            "reduction_kind": reduction_kind,
            "created_at": datetime.now().isoformat(),
        })
        _HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _rf_importance(X: pd.DataFrame, y: pd.Series, informative_numeric_cols: list,
                       max_samples: int = 50_000, n_estimators: int = 50):
        """Quick RandomForest feature importance on informative numeric columns (train-only, sampled).
        Returns (importance_str, importance_df) — df is None on failure."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            cols = [c for c in informative_numeric_cols if c in X.columns]
            if not cols:
                return "(no informative numeric features available)", None
            X_num = X[cols].fillna(0)
            if len(X_num) > max_samples:
                idx = np.random.default_rng(42).choice(len(X_num), max_samples, replace=False)
                X_num = X_num.iloc[idx]
                y_samp = y.iloc[idx]
            else:
                y_samp = y
            if y_samp.dtype == object or hasattr(y_samp.dtype, 'categories'):
                y_enc = LabelEncoder().fit_transform(y_samp.astype(str))
            else:
                y_enc = y_samp.to_numpy()
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=8, n_jobs=-1, random_state=42)
            rf.fit(X_num, y_enc)
            importance_df = pd.DataFrame({'Feature': cols, 'RF Importance': rf.feature_importances_})
            importance_df = importance_df.sort_values('RF Importance', ascending=False).round(6).reset_index(drop=True)
            return importance_df.to_string(index=False), importance_df
        except Exception as e:
            return f"(RF importance computation failed: {e})", None
    @staticmethod
    def _top_corr_pairs(X: pd.DataFrame, importance_df, threshold: float = 0.85,
                        top_n: int = 30, max_pairs: int = 15) -> str:
        """Correlated pairs among top RF-important features; helps avoid redundant evidence seeds."""
        try:
            if importance_df is None or len(importance_df) == 0:
                return "(cannot compute — no importance data)"
            top_feats = importance_df.head(top_n)['Feature'].tolist()
            cols = [c for c in top_feats if c in X.columns]
            if len(cols) < 2:
                return "(fewer than 2 features available)"
            sample = X[cols]
            if len(sample) > 50000:
                sample = sample.sample(n=50000, random_state=42)
            corr = sample.corr(method='pearson').abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = upper.stack().reset_index()
            pairs.columns = ['Feature A', 'Feature B', '|Pearson r|']
            pairs = pairs[pairs['|Pearson r|'] >= threshold].sort_values('|Pearson r|', ascending=False).round(4)
            if pairs.empty:
                return f"(no pairs with |r| ≥ {threshold} among top-{top_n} RF features)"
            if len(pairs) > max_pairs:
                shown = pairs.head(max_pairs).to_string(index=False)
                return f"{shown}\n... [{len(pairs) - max_pairs} more pair(s) hidden]"
            return pairs.to_string(index=False)
        except Exception as e:
            return f"(computation failed: {e})"

    @staticmethod
    def _mutual_info_scores(X: pd.DataFrame, y: pd.Series, informative_cols: list,
                            max_samples: int = 50000, top_n: int = 25) -> str:
        """Mutual information (nonlinear feature-target association) for numeric features."""
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.preprocessing import LabelEncoder
            cols = [c for c in informative_cols if c in X.columns]
            if not cols:
                return "(no informative numeric features available)"
            X_num = X[cols].fillna(0)
            if len(X_num) > max_samples:
                idx = np.random.default_rng(42).choice(len(X_num), max_samples, replace=False)
                X_num = X_num.iloc[idx]
                y_samp = y.iloc[idx]
            else:
                y_samp = y
            if y_samp.dtype == object or hasattr(y_samp.dtype, 'categories'):
                y_enc = LabelEncoder().fit_transform(y_samp.astype(str))
            else:
                y_enc = y_samp.to_numpy()
            mi = mutual_info_classif(X_num, y_enc, random_state=42, n_jobs=-1)
            mi_df = pd.DataFrame({'Feature': cols, 'MI Score': mi})
            mi_df = mi_df.sort_values('MI Score', ascending=False).round(6).reset_index(drop=True)
            return LLM_Suggestions._table_to_prompt(mi_df.head(top_n), top_n)
        except Exception as e:
            return f"(computation failed: {e})"

    @staticmethod
    def _class_feature_means(X: pd.DataFrame, y: pd.Series, importance_df,
                             top_n: int = 10, max_samples: int = 50000, max_classes: int = 10) -> str:
        """Per-class means for top RF features; reveals which features discriminate specific attack types."""
        try:
            if importance_df is None or len(importance_df) == 0:
                return "(cannot compute — no importance data)"
            n_classes = y.nunique()
            if n_classes > max_classes:
                return f"(skipped — {n_classes} classes exceeds display limit of {max_classes})"
            top_feats = importance_df.head(top_n)['Feature'].tolist()
            cols = [c for c in top_feats if c in X.columns]
            if not cols:
                return "(no features available)"
            df = pd.concat([X[cols].reset_index(drop=True), y.rename('__label__').reset_index(drop=True)], axis=1)
            if len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=42)
            means = df.groupby('__label__')[cols].mean().round(4)
            return LLM_Suggestions._table_to_prompt(means.reset_index(), _MAX_FEATURE_ROWS)
        except Exception as e:
            return f"(computation failed: {e})"

    @staticmethod
    def _feature_ks_stability(X_train: pd.DataFrame, X_test, importance_df,
                              top_n: int = 20, max_samples: int = 50000) -> str:
        """KS statistic (train vs test) for top RF features; high KS signals distribution shift."""
        try:
            from scipy.stats import ks_2samp
            if X_test is None or importance_df is None or len(importance_df) == 0:
                return "(cannot compute — test data or importance data not available)"
            top_feats = importance_df.head(top_n)['Feature'].tolist()
            cols = [c for c in top_feats if c in X_train.columns and c in X_test.columns]
            if not cols:
                return "(no common features available)"
            rng = np.random.default_rng(42)
            if len(X_train) > max_samples:
                idx = rng.choice(len(X_train), max_samples, replace=False)
                X_train_s = X_train.iloc[idx]
            else:
                X_train_s = X_train
            if len(X_test) > max_samples:
                idx = rng.choice(len(X_test), max_samples, replace=False)
                X_test_s = X_test.iloc[idx]
            else:
                X_test_s = X_test
            rows = []
            for c in cols:
                stat, _ = ks_2samp(X_train_s[c].dropna().to_numpy(), X_test_s[c].dropna().to_numpy())
                label = "stable" if stat < 0.15 else ("moderate-shift" if stat < 0.30 else "high-shift")
                rows.append({'Feature': c, 'KS Statistic': round(stat, 4), 'Stability': label})
            ks_df = pd.DataFrame(rows).sort_values('KS Statistic', ascending=False)
            return LLM_Suggestions._table_to_prompt(ks_df, _MAX_FEATURE_ROWS)
        except Exception as e:
            return f"(computation failed: {e})"

    @staticmethod
    def _table_to_prompt(df: pd.DataFrame, max_rows: int) -> str:
        """Render a DataFrame for the prompt, capping rows (with a truncation note) to bound tokens."""
        if df is None or len(df) == 0:
            return "(none)"
        if len(df) > max_rows:
            shown = df.head(max_rows).to_string(index=False)
            return f"{shown}\n... [{len(df) - max_rows} more row(s) hidden to conserve context; showing the top {max_rows}]"
        return df.to_string(index=False)

    @staticmethod
    def _build_feature_catalog(X: pd.DataFrame, provided_catalog=None) -> str:
        """Render optional per-dataset feature descriptions for Stage 3.

        Values may be plain description strings or dictionaries containing description,
        aliases, unit, direction, source, and confidence. No keyword glossary is applied:
        an undocumented feature stays explicitly undocumented and remains available to RF/MI.
        """
        if provided_catalog is None:
            provided_catalog = {}
        if not isinstance(provided_catalog, dict):
            raise TypeError("feature_catalog must be None or a dict keyed by exact feature name.")

        rows = []
        for col in X.columns:
            supplied = provided_catalog.get(col)
            if isinstance(supplied, str):
                metadata = {"description": supplied}
            elif isinstance(supplied, dict):
                metadata = supplied
            elif supplied is None:
                metadata = {}
            else:
                raise TypeError(
                    f"feature_catalog[{col!r}] must be a description string or metadata dict."
                )

            description = str(metadata.get("description") or "").strip()
            if description:
                source = str(metadata.get("source") or "provided").strip()
                try:
                    confidence = float(metadata.get("confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 1.0
                confidence = min(1.0, max(0.0, confidence))
            else:
                description = "(not provided — interpret exact name conservatively; keep available to RF/MI)"
                source = "not_provided"
                confidence = 0.0

            aliases = metadata.get("aliases", "")
            if isinstance(aliases, (list, tuple, set)):
                aliases = ", ".join(str(value) for value in aliases)

            rows.append({
                "Feature": col,
                "Dtype": str(X[col].dtype),
                "Cardinality": int(X[col].nunique(dropna=True)),
                "MissingFraction": round(float(X[col].isna().mean()), 6),
                "Description": description,
                "SemanticSource": source,
                "Confidence": round(confidence, 2),
                "Unit": str(metadata.get("unit") or ""),
                "Direction": str(metadata.get("direction") or ""),
                "Aliases": str(aliases or ""),
            })

        return LLM_Suggestions._table_to_prompt(pd.DataFrame(rows), _MAX_FEATURE_ROWS)

    # --- Response validators (used with _call_validated for self-correction) --------------------
    @staticmethod
    def _validate_json(reply: str):
        """(ok, error) — the reply must contain a single strictly-parseable JSON object."""
        value, err = parse_strict_json_object(reply)
        return value is not None, err

    @staticmethod
    def _validate_python(reply: str, stage: str = "stage2"):
        """(ok, error) — the reply's code block must be syntactically valid Python."""
        parsed, err = parse_strict_python_response(reply, stage)
        return parsed is not None, err

    @staticmethod
    def _scale_strategy_to_budget(desired_additions, available_budget):
        available_budget = max(0, int(available_budget))
        total_requested = sum(desired_additions.values())

        if total_requested <= available_budget:
            return desired_additions.copy()

        if total_requested <= 0 or available_budget <= 0:
            return {}

        scale = available_budget / total_requested

        exact_scaled = {
            cls: addition * scale
            for cls, addition in desired_additions.items()
        }
        scaled_additions = {
            cls: int(np.floor(value))
            for cls, value in exact_scaled.items()
        }

        remaining = available_budget - sum(scaled_additions.values())

        allocation_order = sorted(
            desired_additions,
            key=lambda cls: (
                -(exact_scaled[cls] - scaled_additions[cls]),
                str(cls),
            ),
        )

        for cls in allocation_order[:remaining]:
            scaled_additions[cls] += 1

        return {
            cls: addition
            for cls, addition in scaled_additions.items()
            if addition > 0
        }

    @staticmethod
    def _canonicalize_resampling_plan(plan, train_counts, n_rows):
        oversampling_methods = {
            "SMOTE",
            "ADASYN",
            "RandomOverSampler",
            "SMOTETomek",
            "SMOTEENN",
        }
        under_or_none_methods = {
            "None",
            "RandomUnderSampler",
            "TomekLinks",
            "EditedNearestNeighbours",
        }

        counts = {
            class_value: int(count)
            for class_value, count in train_counts.items()
        }

        if not counts:
            return plan

        median_count = float(np.median(list(counts.values())))

        maximum_count = max(counts.values())
        for candidate in plan.get("resampling_candidates", []):
            if not isinstance(candidate, dict):
                continue

            method = str(
                candidate.get("imbalance_handling", "")
            ).strip()

            candidate_id = str(candidate.get("candidate_id", "")).strip()
            if method in under_or_none_methods:
                candidate["alpha"] = None
                candidate["under_fraction"] = None
                candidate["target_class_policy"] = None
                candidate["method_params"] = {}
                candidate["max_expansion_ratio"] = 1.0
                candidate["max_resampled_rows"] = int(n_rows)
                candidate["sampling_strategy_dict"] = None
                if method == "None":
                    continue
                fraction = 0.25 if candidate_id == "mild" else 0.50
                candidate["under_fraction"] = fraction
                targeted = {
                    class_value
                    for class_value, current_count in counts.items()
                    if (
                        current_count == maximum_count
                        if candidate_id == "mild"
                        else current_count > median_count
                    )
                }
                candidate["target_class_policy"] = (
                    "majority_classes" if candidate_id == "mild"
                    else "above_median_classes"
                )
                if method == "RandomUnderSampler":
                    candidate["method_params"] = {
                        "replacement": False, "random_state": 42,
                    }
                    candidate["sampling_strategy_dict"] = {
                        str(class_value): int(np.ceil(
                            current_count
                            - fraction * (current_count - median_count)
                        ))
                        for class_value, current_count in counts.items()
                        if class_value in targeted
                    }
                elif method == "TomekLinks":
                    candidate["method_params"] = {
                        "max_passes": 1 if candidate_id == "mild" else 2,
                    }
                    candidate["sampling_strategy_dict"] = sorted(
                        str(class_value) for class_value in targeted
                    )
                elif method == "EditedNearestNeighbours":
                    candidate["method_params"] = {
                        "n_neighbors": 3,
                        "kind_sel": "mode" if candidate_id == "mild" else "all",
                        "bounded_cleaning_rejected": False,
                    }
                    candidate["sampling_strategy_dict"] = sorted(
                        str(class_value) for class_value in targeted
                    )
                continue

            if method not in oversampling_methods:
                continue

            candidate["under_fraction"] = None
            candidate["target_class_policy"] = "below_half_median"
            candidate["method_params"] = {}
            if candidate_id == "mild":
                candidate["alpha"] = 0.25
                candidate["max_expansion_ratio"] = 1.05
            elif candidate_id == "moderate":
                candidate["alpha"] = 0.50
                candidate["max_expansion_ratio"] = 1.10
            else:
                # Let the structural validator report an invalid candidate id.
                continue

            candidate["max_resampled_rows"] = int(np.floor(
                int(n_rows) * float(candidate["max_expansion_ratio"])
            ))

            try:
                alpha = float(candidate["alpha"])
                max_resampled_rows = int(
                    candidate["max_resampled_rows"]
                )
            except (KeyError, TypeError, ValueError):
                # Let the validator report the precise structural error.
                continue

            available_budget = max_resampled_rows - int(n_rows)
            desired_additions = {}

            for class_value, current_count in counts.items():
                # Only oversample classes below 50% of the median TRAIN class count.
                if current_count >= 0.5 * median_count:
                    continue

                unscaled_target = int(np.ceil(
                    current_count
                    + alpha * (median_count - current_count)
                ))

                addition = max(
                    0,
                    unscaled_target - current_count,
                )

                if addition > 0:
                    desired_additions[class_value] = addition

            scaled_additions = (
                LLM_Suggestions._scale_strategy_to_budget(
                    desired_additions,
                    available_budget,
                )
            )

            candidate["sampling_strategy_dict"] = {
                str(class_value): (
                    counts[class_value] + int(addition)
                )
                for class_value, addition
                in scaled_additions.items()
            }

        return plan

    @staticmethod
    def _validate_stage1_decision(reply: str, all_feature_cols: list,
                                  train_counts: dict, train_features: pd.DataFrame,
                                  verified_drop_metadata: dict | None = None,
                                  parsed_out: dict | None = None):
        """Validate the single resampling-family choice owned by Stage 1."""
        decision, err = parse_strict_json_object(reply)
        if decision is None:
            return False, err
        required = {"resampling_family"}
        if set(decision) != required:
            return False, validation_error(
                "S1_DECISION_FIELDS",
                "Stage 1 must return exactly one resampling_family field; "
                f"missing={sorted(required - set(decision))}, "
                f"extra={sorted(set(decision) - required)}.",
            )

        allowed_methods = {
            "SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler",
            "TomekLinks", "EditedNearestNeighbours", "SMOTETomek", "SMOTEENN", "None",
        }
        method = decision["resampling_family"]
        if not isinstance(method, str) or method not in allowed_methods:
            return False, validation_error(
                "S1_RESAMPLING_FAMILY",
                "resampling_family must be exactly one supported method or 'None'.",
            )

        oversampling_methods = {
            "SMOTE", "ADASYN", "RandomOverSampler", "SMOTETomek", "SMOTEENN",
        }
        synthetic_methods = {"SMOTE", "ADASYN", "SMOTETomek", "SMOTEENN"}
        count_values = [int(value) for value in train_counts.values()]
        median_count = float(np.median(count_values)) if count_values else 0.0
        eligible_counts = [
            count for count in count_values if count < 0.5 * median_count
        ]
        if method in oversampling_methods:
            available_budget = (
                int(np.floor(len(train_features) * 1.05)) - len(train_features)
            )
            if not eligible_counts or available_budget < 1:
                return False, validation_error(
                    "S1_FAMILY_INAPPLICABLE",
                    "The selected oversampling family has no eligible class or mild row budget.",
                )
            if method in synthetic_methods and min(eligible_counts) < 2:
                return False, validation_error(
                    "S1_SYNTHETIC_SUPPORT",
                    "The selected family targets a class below two original samples; "
                    "use RandomOverSampler instead.",
                )

        if parsed_out is not None:
            parsed_out.clear()
            parsed_out.update(copy.deepcopy(decision))
        return True, ""

    @staticmethod
    def _build_host_cleaning_audit(
        train_features: pd.DataFrame,
        verified_drop_metadata: dict | None = None,
    ) -> tuple[list, dict]:
        """Derive conservative cleaning drops without delegating them to Stage 1."""
        drops = []
        evidence = {}
        columns = list(train_features.columns)
        verified = verified_drop_metadata or {}

        for index, column in enumerate(columns):
            series = train_features[column]
            reason = None
            detail = None
            if series.isna().all():
                reason = "all_missing"
                detail = "Every TRAIN value is missing."
            elif series.nunique(dropna=True) <= 1:
                reason = "constant"
                detail = "TRAIN non-missing cardinality is at most one."
            else:
                duplicate_of = next(
                    (
                        previous
                        for previous in columns[:index]
                        if series.equals(train_features[previous])
                    ),
                    None,
                )
                if duplicate_of is not None:
                    reason = "exact_duplicate"
                    detail = f"Exactly duplicates TRAIN feature {duplicate_of!r}."
                else:
                    metadata = verified.get(column, {})
                    verified_reason = metadata.get("verified_drop_reason")
                    verified_evidence = str(metadata.get("drop_evidence", "")).strip()
                    if (
                        verified_reason in {"explicit_leakage", "verified_identifier"}
                        and verified_evidence
                    ):
                        reason = verified_reason
                        detail = verified_evidence
            if reason is not None:
                drops.append(column)
                evidence[column] = {
                    "reason_code": reason,
                    "evidence": detail,
                }
        return drops, evidence

    @staticmethod
    def _build_canonical_plan(decision: dict, train_counts: dict, n_rows: int,
                              categorical_encoding_assignments: dict,
                              null_handling_assignments: dict,
                              host_drop: list | None = None,
                              host_drop_evidence: dict | None = None) -> dict:
        """Merge a validated Stage-1 decision with host-owned deterministic fields."""
        family = decision["resampling_family"]
        count_values = [int(value) for value in train_counts.values()]
        imbalance_ratio = max(count_values) / max(1, min(count_values))
        if imbalance_ratio > 10.0:
            imbalance_severity = "Severe"
        elif imbalance_ratio > 3.0:
            imbalance_severity = "Moderate"
        else:
            imbalance_severity = "Mild"
        family_justification = (
            f"Stage 1 selected the single TRAIN-only family {family}."
        )
        candidates = [{
            "candidate_id": "none",
            "imbalance_handling": "None",
            "justification": "Host reference candidate without resampling.",
        }]
        if family != "None":
            candidates.extend([
                {
                    "candidate_id": "mild",
                    "imbalance_handling": family,
                    "justification": family_justification,
                },
                {
                    "candidate_id": "moderate",
                    "imbalance_handling": family,
                    "justification": family_justification,
                },
            ])
        plan = {
            "rare_class_policy": "keep_all",
            "train_reference_rows": int(n_rows),
            "train_reference_class_counts": {
                str(key): int(value) for key, value in train_counts.items()
            },
            "resampling_selection": {
                "validation_source": "train_derived_validation",
                "splitter": "rare_safe_stratified_holdout",
                "validation_fraction": 0.20,
                "random_state": 42,
                "primary_metric": "weighted_f1",
                "weighted_f1_near_tolerance": 0.003,
                "tie_breakers": ["lower_intensity"],
                "refit_winner_on_full_train": False,
                "test_evaluations": 1,
            },
            "resampling_candidates": candidates,
            "resampling_budget_justification": "Host-owned bounded mild/moderate grid.",
            "downstream_compatibility_justification": "Standard LightGBM selects one shared inner-TRAIN candidate; every final classifier fits only the locked inner-TRAIN winner without a full-TRAIN refit.",
            "imbalance_severity": imbalance_severity,
            "recommended_metrics": ["Weighted F1", "Accuracy", "Per-class Recall"],
            "null_handling": copy.deepcopy(null_handling_assignments),
            "drop": copy.deepcopy(host_drop or []),
            "drop_evidence": copy.deepcopy(host_drop_evidence or {}),
            "categorical_encoding_policy": copy.deepcopy(_CATEGORICAL_ENCODING_POLICY),
            "scaling_policy": copy.deepcopy(_SCALING_POLICY),
            "log_transform_policy": copy.deepcopy(_LOG_TRANSFORM_POLICY),
            "transform": {},
            "encode": copy.deepcopy(categorical_encoding_assignments),
            "normalization": "StandardScaler",
        }
        return LLM_Suggestions._canonicalize_resampling_plan(plan, train_counts, n_rows)

    @staticmethod
    def _validate_plan_json(reply: str, all_feature_cols: list, categorical_feature_cols: list,
                            train_counts: dict, n_rows: int,
                            categorical_cardinalities: dict | None = None,
                            expected_null_handling: dict | None = None):
        """Validate cleaning plus a bounded TRAIN-only resampling candidate set."""
        plan, err = parse_strict_json_object(reply)
        if plan is None:
            return False, err

        required = {
            "rare_class_policy", "resampling_selection",
            "resampling_candidates", "resampling_budget_justification",
            "downstream_compatibility_justification",
            "train_reference_rows", "train_reference_class_counts",
            "drop", "drop_evidence", "encode",
            "categorical_encoding_policy", "scaling_policy",
            "log_transform_policy", "transform", "null_handling",
            "imbalance_severity", "recommended_metrics", "normalization",
        }
        if set(plan) != required:
            return False, validation_error(
                "S1_PLAN_FIELDS",
                f"Canonical PLAN fields must be exact; missing={sorted(required - set(plan))}, "
                f"extra={sorted(set(plan) - required)}.",
            )

        if str(plan["rare_class_policy"]).strip().lower() != "keep_all":
            return False, "rare_class_policy must be 'keep_all', including classes below 1%."
        if plan["imbalance_severity"] not in {"Mild", "Moderate", "Severe"}:
            return False, "imbalance_severity must be Mild, Moderate, or Severe."
        if plan["recommended_metrics"] != [
            "Weighted F1", "Accuracy", "Per-class Recall"
        ]:
            return False, "recommended_metrics must equal the host-owned metric audit."
        if plan["normalization"] != "StandardScaler":
            return False, "normalization must equal 'StandardScaler'."
        if plan["categorical_encoding_policy"] != _CATEGORICAL_ENCODING_POLICY:
            return False, (
                "categorical_encoding_policy must exactly equal the deterministic "
                f"cross-fitted Target Encoding contract: {_CATEGORICAL_ENCODING_POLICY}."
            )
        if plan["scaling_policy"] != _SCALING_POLICY:
            return False, (
                "scaling_policy must exactly equal the deterministic TRAIN-only "
                f"StandardScaler contract: {_SCALING_POLICY}."
            )
        if plan["log_transform_policy"] != _LOG_TRANSFORM_POLICY:
            return False, (
                "log_transform_policy must exactly equal the deterministic "
                f"TRAIN-only policy: {_LOG_TRANSFORM_POLICY}."
            )
        if plan["transform"] != {}:
            return False, (
                "transform must be an empty object; Stage 2 derives log_transform_cols "
                "from TRAIN values and feature-name transform lists are forbidden."
            )
        if plan["train_reference_rows"] != int(n_rows):
            return False, f"train_reference_rows must equal the supplied TRAIN row count {int(n_rows)}."
        reference_counts = plan["train_reference_class_counts"]
        if not isinstance(reference_counts, dict) or len(reference_counts) != len(train_counts):
            return False, "train_reference_class_counts must copy every supplied TRAIN class count exactly."
        resolved_reference_counts = {}
        for plan_key, count in reference_counts.items():
            matches = [existing for existing in train_counts if str(existing) == str(plan_key)]
            if len(matches) != 1 or isinstance(count, bool) or not isinstance(count, int):
                return False, (f"Invalid or ambiguous TRAIN reference class/count entry: "
                               f"{plan_key!r}: {count!r}.")
            resolved_reference_counts[matches[0]] = count
        if resolved_reference_counts != {key: int(value) for key, value in train_counts.items()}:
            return False, "train_reference_class_counts does not exactly match the supplied TRAIN counts."

        selection = plan["resampling_selection"]
        if not isinstance(selection, dict):
            return False, "resampling_selection must be an object."
        expected_selection_fields = {
            "validation_source", "splitter", "validation_fraction", "random_state",
            "primary_metric", "weighted_f1_near_tolerance", "tie_breakers",
            "refit_winner_on_full_train", "test_evaluations",
        }
        if set(selection) != expected_selection_fields:
            return False, validation_error(
                "S1_SELECTION_FIELDS",
                "resampling_selection fields must exactly match the host contract.",
            )
        if str(selection.get("validation_source", "")).strip().lower() != "train_derived_validation":
            return False, "resampling_selection.validation_source must be 'train_derived_validation'."
        if selection.get("splitter") != "rare_safe_stratified_holdout":
            return False, "resampling_selection.splitter must be 'rare_safe_stratified_holdout'."
        try:
            validation_fraction = float(selection.get("validation_fraction"))
        except (TypeError, ValueError):
            return False, "resampling_selection.validation_fraction must be numeric."
        if not np.isclose(validation_fraction, 0.20):
            return False, "resampling_selection.validation_fraction must equal 0.20."
        if selection.get("random_state") != 42:
            return False, "resampling_selection.random_state must equal 42."
        if str(selection.get("primary_metric", "")).strip().lower() != "weighted_f1":
            return False, "resampling_selection.primary_metric must be 'weighted_f1'."
        try:
            weighted_f1_near_tolerance = float(selection.get("weighted_f1_near_tolerance"))
        except (TypeError, ValueError):
            return False, "weighted_f1_near_tolerance must be numeric."
        if not np.isclose(weighted_f1_near_tolerance, 0.003):
            return False, "weighted_f1_near_tolerance must equal 0.003."
        expected_tie_breakers = ["lower_intensity"]
        if selection.get("tie_breakers") != expected_tie_breakers:
            return False, ("resampling_selection.tie_breakers must equal "
                           f"{expected_tie_breakers} for deterministic candidate selection.")
        if selection.get("refit_winner_on_full_train") is not False or selection.get("test_evaluations") != 1:
            return False, ("The winning candidate must remain fitted on the locked inner TRAIN without "
                           "a full-TRAIN refit, and held-out TEST must be evaluated exactly once.")

        allowed_methods = {
            "SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler", "TomekLinks",
            "EditedNearestNeighbours", "SMOTETomek", "SMOTEENN", "None",
        }
        oversampling_methods = {"SMOTE", "ADASYN", "RandomOverSampler", "SMOTETomek", "SMOTEENN"}
        under_or_none = {"None", "RandomUnderSampler", "TomekLinks", "EditedNearestNeighbours"}
        candidates = plan["resampling_candidates"]
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
            return False, "resampling_candidates must contain between one and three candidate objects."
        candidate_ids = [item.get("candidate_id") for item in candidates if isinstance(item, dict)]
        if len(candidate_ids) != len(candidates) or any(not isinstance(value, str) or not value for value in candidate_ids):
            return False, "Every resampling candidate must define a non-empty string candidate_id."
        if len(candidate_ids) != len(set(candidate_ids)):
            return False, "resampling candidate_id values must be unique."
        unknown_candidate_ids = sorted(set(candidate_ids) - {"none", "mild", "moderate"})
        if unknown_candidate_ids:
            return False, ("resampling candidate_id values are limited to 'none', 'mild', and "
                           f"'moderate'; received: {unknown_candidate_ids}.")
        if "none" not in candidate_ids:
            return False, "resampling_candidates must always include candidate_id='none'."
        if candidate_ids not in (["none"], ["none", "mild", "moderate"]):
            return False, (
                "The host grid must be exactly ['none'] or "
                "['none', 'mild', 'moderate'] in that order."
            )
        if len(candidates) == 3:
            mild_method = candidates[1].get("imbalance_handling")
            moderate_method = candidates[2].get("imbalance_handling")
            if mild_method == "None" or mild_method != moderate_method:
                return False, validation_error(
                    "S1_SINGLE_FAMILY", "mild and moderate must use the same resampling family."
                )

        majority_count = max(int(value) for value in train_counts.values())
        median_count = float(np.median([int(value) for value in train_counts.values()]))
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            required_candidate_fields = {
                "candidate_id", "imbalance_handling", "alpha", "under_fraction",
                "target_class_policy", "method_params", "max_expansion_ratio",
                "max_resampled_rows", "sampling_strategy_dict", "justification",
            }
            missing_candidate = sorted(required_candidate_fields - set(candidate))
            extra_candidate = sorted(set(candidate) - required_candidate_fields)
            if missing_candidate or extra_candidate:
                return False, (
                    f"Candidate {candidate_id!r} fields must be exact; "
                    f"missing={missing_candidate}, extra={extra_candidate}."
                )

            method = str(candidate["imbalance_handling"]).strip()
            if method not in allowed_methods:
                return False, (f"Candidate {candidate_id!r} imbalance_handling must be one of "
                               f"{sorted(allowed_methods)}.")
            if not str(candidate["justification"]).strip():
                return False, f"Candidate {candidate_id!r} justification must be non-empty."

            if candidate_id == "none" and method != "None":
                return False, "candidate_id='none' must use imbalance_handling='None'."
            if method == "None" and candidate_id != "none":
                return False, "Only candidate_id='none' may use imbalance_handling='None'."

            try:
                ratio = float(candidate["max_expansion_ratio"])
            except (TypeError, ValueError):
                return False, f"Candidate {candidate_id!r} max_expansion_ratio must be numeric."
            if method in under_or_none:
                expected_ratio = 1.0
            else:
                expected_ratio = 1.05 if candidate_id == "mild" else 1.10
                if candidate_id not in {"mild", "moderate"}:
                    return False, "Oversampling candidate_id must be 'mild' or 'moderate'."
            if not np.isclose(ratio, expected_ratio):
                return False, (f"Candidate {candidate_id!r} max_expansion_ratio must be "
                               f"{expected_ratio:.2f}; received {ratio!r}.")

            max_rows = candidate["max_resampled_rows"]
            expected_max_rows = int(np.floor(n_rows * expected_ratio))
            if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows != expected_max_rows:
                return False, (f"Candidate {candidate_id!r} max_resampled_rows must equal "
                               f"floor({n_rows} * {expected_ratio:.2f}) = {expected_max_rows}.")

            strategy = candidate["sampling_strategy_dict"]
            if method in oversampling_methods:
                expected_alpha = 0.25 if candidate_id == "mild" else 0.50
                try:
                    alpha = float(candidate["alpha"])
                except (TypeError, ValueError):
                    return False, f"Candidate {candidate_id!r} alpha must be numeric."
                if not np.isclose(alpha, expected_alpha):
                    return False, f"Candidate {candidate_id!r} alpha must equal {expected_alpha:.2f}."
                if candidate["under_fraction"] is not None:
                    return False, f"Candidate {candidate_id!r} under_fraction must be null for oversampling."
                if candidate["target_class_policy"] != "below_half_median":
                    return False, f"Candidate {candidate_id!r} must target only classes below half the median."
                if candidate["method_params"] != {}:
                    return False, f"Candidate {candidate_id!r} method_params must be empty for {method!r}."
                if not isinstance(strategy, dict) or not strategy:
                    return False, f"Candidate {candidate_id!r} requires an explicit sampling_strategy_dict."

                resolved_strategy = {}
                for plan_key, target_count in strategy.items():
                    matches = [existing for existing in train_counts if str(existing) == str(plan_key)]
                    if len(matches) != 1:
                        return False, (f"Candidate {candidate_id!r} class key {plan_key!r} does not "
                                       "resolve uniquely to an existing TRAIN class.")
                    class_value = matches[0]
                    if isinstance(target_count, bool) or not isinstance(target_count, int):
                        return False, f"Candidate {candidate_id!r} target for {plan_key!r} must be an integer."
                    current_count = int(train_counts[class_value])
                    if current_count >= 0.5 * median_count:
                        return False, (f"Candidate {candidate_id!r} targets class {plan_key!r}, but its "
                                       "TRAIN count is not below 0.5 * median TRAIN class count.")
                    unscaled_ceiling = int(np.ceil(current_count + alpha * (median_count - current_count)))
                    if target_count <= current_count or target_count > unscaled_ceiling:
                        return False, (f"Candidate {candidate_id!r} target for {plan_key!r} must be above "
                                       f"{current_count} and at most the shrinkage target {unscaled_ceiling}.")
                    if target_count >= majority_count:
                        return False, f"Candidate {candidate_id!r} may not match the TRAIN majority."
                    resolved_strategy[class_value] = target_count

                projected_rows = n_rows + sum(
                    target_count - int(train_counts[class_value])
                    for class_value, target_count in resolved_strategy.items()
                )
                if projected_rows > max_rows:
                    return False, (f"Candidate {candidate_id!r} projects {projected_rows} TRAIN rows, "
                                   f"exceeding the hard budget {max_rows}.")
                if method in {"SMOTE", "ADASYN", "SMOTETomek", "SMOTEENN"}:
                    smallest_targeted = min(int(train_counts[class_value]) for class_value in resolved_strategy)
                    if smallest_targeted < 2:
                        return False, ("Synthetic-neighbor sampling targets a class with fewer than 2 TRAIN "
                                       "samples; use RandomOverSampler without dropping the class.")
            else:
                if candidate["alpha"] is not None:
                    return False, f"Candidate {candidate_id!r} alpha must be null for method {method!r}."
                if method == "None":
                    if any((
                        candidate["under_fraction"] is not None,
                        candidate["target_class_policy"] is not None,
                        candidate["method_params"] != {},
                        strategy is not None,
                    )):
                        return False, "The none candidate must not contain resampling parameters."
                    continue
                expected_fraction = 0.25 if candidate_id == "mild" else 0.50
                try:
                    under_fraction = float(candidate["under_fraction"])
                except (TypeError, ValueError):
                    return False, f"Candidate {candidate_id!r} under_fraction must be numeric."
                if not np.isclose(under_fraction, expected_fraction):
                    return False, f"Candidate {candidate_id!r} has an incorrect under_fraction."
                expected_policy = (
                    "majority_classes" if candidate_id == "mild"
                    else "above_median_classes"
                )
                if candidate["target_class_policy"] != expected_policy:
                    return False, f"Candidate {candidate_id!r} has an incorrect target_class_policy."
                targeted = {
                    str(class_value)
                    for class_value, current_count in train_counts.items()
                    if (
                        int(current_count) == majority_count
                        if candidate_id == "mild"
                        else int(current_count) > median_count
                    )
                }
                if method == "RandomUnderSampler":
                    expected_strategy = {
                        str(class_value): int(np.ceil(
                            int(current_count)
                            - expected_fraction * (int(current_count) - median_count)
                        ))
                        for class_value, current_count in train_counts.items()
                        if str(class_value) in targeted
                    }
                    if strategy != expected_strategy:
                        return False, f"Candidate {candidate_id!r} has incorrect RandomUnderSampler floors."
                    if candidate["method_params"] != {"replacement": False, "random_state": 42}:
                        return False, f"Candidate {candidate_id!r} has incorrect RandomUnderSampler parameters."
                elif method == "TomekLinks":
                    expected_params = {"max_passes": 1 if candidate_id == "mild" else 2}
                    if set(strategy or []) != targeted or candidate["method_params"] != expected_params:
                        return False, f"Candidate {candidate_id!r} has incorrect TomekLinks bounds."
                elif method == "EditedNearestNeighbours":
                    expected_params = {
                        "n_neighbors": 3,
                        "kind_sel": "mode" if candidate_id == "mild" else "all",
                        "bounded_cleaning_rejected": False,
                    }
                    if set(strategy or []) != targeted or candidate["method_params"] != expected_params:
                        return False, f"Candidate {candidate_id!r} has incorrect ENN bounds."

        if not str(plan["resampling_budget_justification"]).strip():
            return False, "resampling_budget_justification must be non-empty."
        if not str(plan["downstream_compatibility_justification"]).strip():
            return False, "downstream_compatibility_justification must be non-empty."

        drop = plan["drop"]
        evidence = plan["drop_evidence"]
        if not isinstance(drop, list) or not isinstance(evidence, dict):
            return False, "drop must be a list and drop_evidence must be an object."
        if len(drop) != len(set(drop)) or not set(drop).issubset(set(all_feature_cols)):
            return False, "drop must contain unique, exact feature names from the supplied feature list."
        if set(evidence) != set(drop):
            return False, "Every dropped feature, and only a dropped feature, must have drop_evidence."
        allowed_reasons = {"constant", "all_missing", "exact_duplicate", "explicit_leakage", "verified_identifier"}
        for col, item in evidence.items():
            if not isinstance(item, dict) or item.get("reason_code") not in allowed_reasons or not str(item.get("evidence", "")).strip():
                return False, (f"Invalid drop_evidence for {col!r}; keyword/name-only rules are forbidden and "
                               f"reason_code must be one of {sorted(allowed_reasons)} with concrete evidence.")

        null_handling = plan["null_handling"]
        if not isinstance(null_handling, dict):
            return False, "null_handling must be an object."
        if expected_null_handling is not None and null_handling != expected_null_handling:
            return False, (
                "null_handling must exactly equal the deterministic TRAIN-only "
                f"imputation audit: {expected_null_handling!r}."
            )
        allowed_null_methods = {"median", "mode", "not_applicable_dropped"}
        invalid_null_methods = {
            col: method
            for col, method in null_handling.items()
            if method not in allowed_null_methods
        }
        if invalid_null_methods:
            return False, (
                "null_handling contains unsupported methods; row dropping and "
                f"ambiguous fallbacks are forbidden: {invalid_null_methods!r}."
            )
        all_missing_expected = {
            col
            for col, method in null_handling.items()
            if method == "not_applicable_dropped"
        }
        if not all_missing_expected.issubset(set(drop)):
            return False, (
                "Every all-missing feature marked not_applicable_dropped must also "
                "appear in drop with all_missing evidence."
            )
        for col in sorted(all_missing_expected):
            item = evidence.get(col)
            if not isinstance(item, dict) or item.get("reason_code") != "all_missing":
                return False, (
                    f"All-missing feature {col!r} must use drop_evidence "
                    "reason_code='all_missing'."
                )

        encode = plan["encode"]
        if not isinstance(encode, dict):
            return False, (
                "encode must map every original categorical feature to its exact "
                "TRAIN cardinality and policy-derived method."
            )
        expected_categorical_keys = set(categorical_feature_cols)
        if set(encode) != expected_categorical_keys:
            missing_categoricals = sorted(expected_categorical_keys - set(encode))
            extra_categoricals = sorted(set(encode) - expected_categorical_keys)
            return False, (
                "encode must list every original categorical feature exactly once; "
                f"missing={missing_categoricals}, extra={extra_categoricals}."
            )
        if categorical_cardinalities is not None:
            for col in categorical_feature_cols:
                cardinality = categorical_cardinalities.get(col)
                if isinstance(cardinality, bool) or not isinstance(cardinality, int):
                    return False, f"Missing exact TRAIN cardinality for categorical feature {col!r}."
                if cardinality < 2:
                    if col not in drop:
                        return False, (
                            f"Categorical feature {col!r} has TRAIN cardinality {cardinality}; "
                            "it must be dropped with verified constant/all-missing evidence."
                        )
                    # The entry remains an audit record but is never executed after the drop.
                    expected_method = "not_applicable_dropped"
                else:
                    expected_method = _categorical_encoding_method(cardinality)
                expected_entry = {
                    "train_cardinality": cardinality,
                    "method": expected_method,
                }
                if encode[col] != expected_entry:
                    return False, (
                        f"encode[{col!r}] must be exactly {expected_entry!r}; "
                        f"received {encode[col]!r}."
                    )
        return True, ""

    @staticmethod
    def _validate_preprocessor_code(reply: str, contract_train: pd.DataFrame,
                                    contract_test: pd.DataFrame, contract_y: pd.Series,
                                    approved_plan: dict):
        return validate_preprocessor_response(
            reply, contract_train, contract_test, contract_y, approved_plan
        )

    @staticmethod
    def _validate_reducer_code(reply: str, numeric_columns: list | None = None,
                               categorical_columns: list | None = None):
        return validate_reducer_response(
            reply, numeric_columns, categorical_columns
        )


    def _call_validated(self, prompt: str, system_prompt: str, validator, *,
                        temperature: float, label: str, max_retries: int = 1,
                        rejection_log_dir: Path | None = None, base_seed: int = 42) -> str:
        """Call the LLM, validate it, and request a fresh complete response on failure.

        `validator(reply) -> (ok: bool, error_message: str)`. Returns only a validated reply.
        Invalid code is never loaded by the pipeline. When rejection_log_dir is supplied,
        rejected text and diagnostics are retained there for audit only. Each semantic
        attempt uses reset=True and a distinct seed; provider failures reuse the same
        prompt/seed without consuming a semantic attempt."""
        attempt_prompt = prompt
        reply = ""
        err = "No response was validated."
        feedback_errors = []
        for attempt in range(max_retries + 1):
            attempt_seed = int(base_seed) + attempt
            attempt_temperature = (
                float(temperature) if attempt == 0 else max(float(temperature), 0.1)
            )
            # Empty/non-text replies are transport/provider failures, not semantic
            # correction attempts. Retry them before consuming this attempt.
            reply = None
            for empty_retry in range(3):
                reply = self.api_caller.call_llm(
                    attempt_prompt,
                    system_prompt,
                    temperature=attempt_temperature,
                    reset=True,
                    seed=attempt_seed,
                )
                if isinstance(reply, str) and reply.strip():
                    break
                print(
                    f"[LLM] {label}: empty/non-text response "
                    f"(transport retry {empty_retry + 1}/3)."
                )

            if not isinstance(reply, str) or not reply.strip():
                had_text_response = False
                ok = False
                err = (
                    "The provider returned no textual content after 3 transport retries. "
                    "The response may have been truncated, refused, or contained only "
                    "non-text content blocks."
                )
            else:
                had_text_response = True
                try:
                    ok, err = validator(reply)
                except Exception as exc:
                    ok = False
                    err = validation_error(
                        "VALIDATOR_INTERNAL",
                        f"The validator raised {type(exc).__name__}: {str(exc)[:300]}",
                    )

            err = str(err or "The response failed validation without an error message.")
            if ok:
                return reply
            print(
                f"[LLM] {label}: response rejected "
                f"(attempt {attempt + 1}/{max_retries + 1}, "
                f"seed={attempt_seed}, temperature={attempt_temperature:g}) — {err[:160]}"
            )
            if had_text_response:
                if rejection_log_dir is not None:
                    rejection_log_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"attempt_{attempt + 1}"
                    (rejection_log_dir / f"{stem}_response.txt").write_text(
                        reply, encoding="utf-8"
                    )
                    (rejection_log_dir / f"{stem}_error.txt").write_text(
                        err + "\n", encoding="utf-8"
                    )
                for line in (item.strip() for item in err.splitlines()):
                    if not line:
                        continue
                    if not re.match(r"^\[[A-Z0-9_]+\]", line):
                        line = validation_error("VALIDATION_CONTRACT", line)
                    if line not in feedback_errors:
                        feedback_errors.append(line)
                if "TargetEncoder" in err and "width" in err.lower():
                    repair = validation_error(
                        "S2_TARGET_WIDTH_REPAIR",
                        "TargetEncoder output width equals the number of original TRAIN "
                        "target classes, never source cardinality. Copy the canonical "
                        "high-cardinality branch and preserve its detailed diagnostics.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if "S2_UNAPPROVED_SAMPLER" in err:
                    repair = validation_error(
                        "S2_UNAPPROVED_SAMPLER_REPAIR",
                        "Delete every import, constructor, helper, and dead branch for "
                        "sampler families other than the single family approved in "
                        "the original prompt. Do not build a generic sampler factory.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if (
                    "S2_METADATA_HARDCODE" in err
                    or "Unexpected TRAIN_REFERENCE_ROWS" in err
                    or "Unexpected TRAIN_REFERENCE_CLASS_COUNTS" in err
                ):
                    repair = validation_error(
                        "S2_METADATA_HARDCODE_REPAIR",
                        "Remove literal guards for "
                        "RESAMPLING_CANDIDATES, TRAIN_REFERENCE_ROWS, and "
                        "TRAIN_REFERENCE_CLASS_COUNTS. Candidate validation substitutes "
                        "scaled equivalents; consume current values.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if (
                    "S2_CANDIDATE_RUNTIME" in err
                    and "KeyError:" in err
                    and any(
                        f"'{candidate_id}'" in err
                        for candidate_id in ("none", "mild", "moderate")
                    )
                ):
                    repair = validation_error(
                        "S2_CANDIDATE_SIBLING_REPAIR",
                        "Never index a sibling candidate by a fixed ID. Validate the "
                        "family across the non-none candidates currently present; "
                        "substituted metadata may contain only none plus one candidate.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if "S2_PY_TRUNCATED" in err:
                    repair = validation_error(
                        "S2_TRUNCATION_REPAIR",
                        "Regenerate the entire module in one Python fence. Keep it "
                        "short: omit explanatory comments and redundant self-audits; "
                        "do not continue the truncated text.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if "Cannot setitem on a Categorical with a new category" in err:
                    repair = validation_error(
                        "S2_CATEGORICAL_DTYPE_REPAIR",
                        "Inside the generic raw_name branch, create independent "
                        "train_values and test_values with astype('string') before "
                        "TRAIN-mode fillna. Never fill original category-dtype columns.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if "OHE source" in err and "active child" in err:
                    repair = validation_error(
                        "S2_OHE_LINEAGE_REPAIR",
                        "Store ohe_children_by_source[raw_name] from "
                        "get_feature_names_out only inside the actual current-runtime "
                        "OHE branch and audit that mapping directly. Do not add "
                        "binary/TargetEncoder children or gate this audit on PLAN "
                        "method/cardinality. Never use split or rsplit on child names.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if "S2_LOG_SELECTION" in err:
                    repair = validation_error(
                        "S2_LOG_SELECTION_REPAIR",
                        "Compute log1p eligibility after only numeric coercion, "
                        "Inf-to-NaN replacement, and TRAIN-median filling, before "
                        "clipping, winsorization, capping, scaling, or resampling.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if (
                    "S3_RUNTIME_EXCEPTION" in err
                    and "Runtime audit concatenation must match dataframe column order exactly" in err
                ):
                    repair = validation_error(
                        "S3_AUDIT_ORDER_REPAIR",
                        "Encoding-audit lists are a disjoint set partition of the runtime "
                        "schema; their concatenation does not have to reproduce DataFrame "
                        "column order. Validate disjointness and set equality only, while "
                        "preserving the original DataFrame order for TRAIN/TEST outputs.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if (
                    "S3_RUNTIME_EXCEPTION" in err
                    and (
                        "Too many semantic candidates" in err
                        or "Too many protected candidates" in err
                    )
                ):
                    repair = validation_error(
                        "S3_SEMANTIC_SELF_LIMIT_REPAIR",
                        "Keep the original tier contract: nominate at most 20 semantic "
                        "raw candidates and at most 10 protected tier-0 candidates. "
                        "Do not raise a runtime self-limit error after emitting lists "
                        "that already satisfy those bounds.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if "S3_COLUMN_SCOPE" in err:
                    repair = validation_error(
                        "S3_COLUMN_SCOPE_REPAIR",
                        "Keep one representative per raw group only in the semantic "
                        "seed. Score every runtime column independently during RF+MI "
                        "completion, so additional qualified encoded siblings may still "
                        "enter the final feature set.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if "S3_SEMANTIC_REPRESENTATIVE" in err:
                    repair = validation_error(
                        "S3_SEMANTIC_REPRESENTATIVE_REPAIR",
                        "Build the semantic seed by raw group. Sum semantic RF child "
                        "importances to rank groups, keep only the highest-importance "
                        "runtime descendant as each selected group's representative, "
                        "and align semantic_seed_raw_groups one-to-one with "
                        "semantic_seed_columns.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if "S3_PROTECTED_ANCHOR" in err:
                    repair = validation_error(
                        "S3_PROTECTED_ANCHOR_REPAIR",
                        "Record protected_semantic_anchor_columns as the ordered "
                        "semantic representatives whose raw groups are protected. Every "
                        "semantic group, protected or not, contributes at most one seed "
                        "column; siblings remain eligible for RF+MI completion.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if "S3_SEMANTIC_LINEAGE" in err:
                    repair = validation_error(
                        "S3_SEMANTIC_LINEAGE_REPAIR",
                        "Resolve each categorical semantic raw group to every runtime "
                        "column with its raw-name prefix across OHE, target-encoded, "
                        "and binary audit types. Record the complete pre-pruning pool "
                        "in resolved_semantic_columns; do not filter lineage to OHE.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                if "S3_SAFE_MODULE_SCOPE" in err or (
                    "S3_API" in err and "REDUCTION_KIND" in err
                ):
                    repair = validation_error(
                        "S3_MODULE_SCOPE_REPAIR",
                        "Keep REDUCTION_KIND = 'hybrid' at module scope. Semantic "
                        "literals and the canonical tier-0 list comprehension may also "
                        "remain there; move every other dynamic assignment inside "
                        "reduce_features or a helper.",
                    )
                    if repair not in feedback_errors:
                        feedback_errors.append(repair)
                attempt_prompt = f"{prompt}\n\n{format_feedback(feedback_errors)}"
            else:
                # A provider/transport failure offers no semantic diagnostic.
                attempt_prompt = prompt
        raise RuntimeError(
            f"[LLM] {label}: no valid response after {max_retries + 1} attempt(s). "
            f"Last validation error: {err}"
        )

    def generate(self, train_df, label_col_name, dataset_name, classification_type: str = "multiclass",
                 test_df=None, prior_summary: str = "", feature_catalog=None, generation_seed: int = 42):
        if classification_type != "multiclass":
            raise ValueError("Only multiclass LLM generation is supported.")
        import random
        random.seed(generation_seed)
        np.random.seed(generation_seed)
        generation_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        if not isinstance(train_df, pd.DataFrame) or train_df.empty:
            raise ValueError("Stage generation requires a non-empty TRAIN DataFrame.")
        if label_col_name not in train_df.columns:
            raise KeyError(f"Target column {label_col_name!r} is missing from TRAIN.")
        if train_df.columns.duplicated().any():
            duplicates = train_df.columns[train_df.columns.duplicated()].tolist()
            raise ValueError(f"TRAIN contains duplicate column names: {duplicates!r}.")
        if train_df[label_col_name].isna().any():
            raise ValueError("TRAIN target contains missing labels; format the target in the prior stage.")
        if train_df[label_col_name].nunique(dropna=True) < 2:
            raise ValueError("Classification requires at least two TRAIN target classes.")

        # Prepare feature statistics. reset_index() keeps the feature NAME as a visible 'feature'
        # column (describe().T puts it in the index, which to_string(index=False) would otherwise hide).
        summary_df = train_df.describe(include='all').T
        summary_df.index.name = 'feature'
        summary_df = summary_df.reset_index()
        _stat_num_cols = summary_df.select_dtypes(include='number').columns
        summary_df[_stat_num_cols] = summary_df[_stat_num_cols].round(4)

        # Column typing (target excluded from feature lists)
        numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = train_df.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()
        numeric_feature_cols = [c for c in numeric_cols if c != label_col_name]
        categorical_feature_cols = [c for c in categorical_cols if c != label_col_name]
        categorical_cardinalities = {
            col: int(train_df[col].nunique(dropna=True))
            for col in categorical_feature_cols
        }
        categorical_encoding_assignments = {
            col: {
                "train_cardinality": cardinality,
                "method": (
                    _categorical_encoding_method(cardinality)
                    if cardinality >= 2
                    else "not_applicable_dropped"
                ),
            }
            for col, cardinality in categorical_cardinalities.items()
        }
        feature_cols = numeric_feature_cols + categorical_feature_cols
        null_handling_assignments = {}
        for col in feature_cols:
            missing_count = int(train_df[col].isna().sum())
            if missing_count == 0:
                continue
            if int(train_df[col].notna().sum()) == 0:
                null_handling_assignments[col] = "not_applicable_dropped"
            elif col in categorical_feature_cols:
                null_handling_assignments[col] = "mode"
            else:
                null_handling_assignments[col] = "median"

        # Near-zero-variance numeric features (constant / std≈0). Surfaced compactly as a drop-hint and
        # excluded from the (potentially large) correlation matrix to save context tokens and NaN clutter.
        nzv_features = [
            c for c in numeric_feature_cols
            if train_df[c].nunique(dropna=True) <= 1 or float(train_df[c].std(skipna=True) or 0.0) <= _NZV_STD_EPS
        ]
        informative_numeric_cols = [c for c in numeric_feature_cols if c not in nzv_features]

        # 1. Numeric Feature-Feature Correlation (Pearson, keep pairs >= 0.85, no duplicates)
        # np.triu (upper triangle) removes the main diagonal (corr=1) and the symmetric lower half.
        num_corr_matrix = train_df[informative_numeric_cols].corr().abs()
        upper_tri = num_corr_matrix.where(np.triu(np.ones(num_corr_matrix.shape), k=1).astype(bool))
        correlation_df = upper_tri.stack().reset_index()
        correlation_df.columns = ['Feature 1', 'Feature 2', 'Absolute Correlation']
        correlation_df = correlation_df[correlation_df['Absolute Correlation'] >= 0.85]
        correlation_df = correlation_df.sort_values(by='Absolute Correlation', ascending=False).round(4)

        # 2. Categorical Feature-Feature Association (Chi-square → Cramér's V, all pairs)
        cat_corr_rows = []
        for i in range(len(categorical_feature_cols)):
            for j in range(i + 1, len(categorical_feature_cols)):
                a, b = categorical_feature_cols[i], categorical_feature_cols[j]
                if train_df[a].nunique() > _MAX_CARD_FOR_CHI2 or train_df[b].nunique() > _MAX_CARD_FOR_CHI2:
                    continue
                cat_corr_rows.append({
                    'Feature 1': a, 'Feature 2': b,
                    "Cramér's V": _cramers_v(train_df[a], train_df[b]),
                    'p-value': _chi2_pvalue(train_df[a], train_df[b]),
                })
        cat_correlation_df = pd.DataFrame(cat_corr_rows, columns=['Feature 1', 'Feature 2', "Cramér's V", 'p-value'])
        if not cat_correlation_df.empty:
            cat_correlation_df = cat_correlation_df.sort_values(by="Cramér's V", ascending=False, na_position='last').round(4)

        # 3. Type-aware feature-target association (treat the target as categorical)
        ft_rows = []
        for col in train_df.columns:
            if col == label_col_name:
                continue
            feat_is_num = col in numeric_cols

            if feat_is_num:
                # Numeric feature vs. categorical target: use ANOVA.
                method = "ANOVA (F-test)"
                assoc, pval = _anova_f(train_df[col], train_df[label_col_name])
            else:
                # Categorical feature vs. categorical target: use chi-square.
                method = "Chi-square (Cramér's V)"
                if train_df[col].nunique() > _MAX_CARD_FOR_CHI2 or train_df[label_col_name].nunique() > _MAX_CARD_FOR_CHI2:
                    assoc, pval = float('nan'), float('nan')
                else:
                    assoc = _cramers_v(train_df[col], train_df[label_col_name])
                    pval = _chi2_pvalue(train_df[col], train_df[label_col_name])

            ft_rows.append({
                'Feature': col,
                'Type': 'numeric' if feat_is_num else 'categorical',
                'Method': method,
                'Association': assoc,
                'p-value': pval,
            })
        target_correlation_df = pd.DataFrame(ft_rows, columns=['Feature', 'Type', 'Method', 'Association', 'p-value'])
        if not target_correlation_df.empty:
            target_correlation_df = target_correlation_df.sort_values(
                by=['Method', 'Association'], ascending=[True, False], na_position='last'
            ).round(4)

        # 4. Target class distribution from TRAIN only (absolute counts & proportions)
        if label_col_name in train_df.columns:
            train_counts = train_df[label_col_name].value_counts().to_dict()
            train_props = train_df[label_col_name].value_counts(normalize=True).round(6).to_dict()
            class_distribution = f"Absolute Counts: {train_counts}\n  Proportions: {train_props}"
        else:
            class_distribution = "Not Available"

        # 5. Dataset scale metadata (drives the imbalance scale/scarcity rules)
        n_rows = len(train_df)

        # Feature-only TRAIN matrix used for Stage-2 contract fixtures and the optional
        # Stage-3 feature catalog. Stage 3 refits its own RF/MI evidence at runtime, so
        # precomputing duplicate RF/MI/KS context here is unnecessary. In particular,
        # held-out TEST feature distributions are never inspected during generation.
        X_for_rf = train_df.drop(columns=[label_col_name], errors='ignore')

        # Render the reference tables once, capped + truncation-noted, to bound prompt size (context
        # budget). Pairwise tables are quadratic ⇒ tighter cap; per-feature tables get a larger cap.
        summary_str = self._table_to_prompt(summary_df, _MAX_FEATURE_ROWS)
        numcorr_str = self._table_to_prompt(correlation_df, _MAX_PAIR_ROWS)
        catcorr_str = self._table_to_prompt(cat_correlation_df, _MAX_PAIR_ROWS)
        fts_str = self._table_to_prompt(target_correlation_df, _MAX_FEATURE_ROWS)
        nzv_note = str(nzv_features) if nzv_features else "(none detected)"

        base_context = build_base_context()
        dataset_context = build_dataset_context(dataset_name, prior_summary)
        analyst_system_prompt = build_stage1_system_prompt(base_context)
        engineer_system_prompt = build_stage2_system_prompt(base_context)
        feature_system_prompt = build_stage3_system_prompt(base_context)

        catalog_for_dataset = feature_catalog
        if (
            isinstance(feature_catalog, dict)
            and dataset_name in feature_catalog
            and isinstance(feature_catalog[dataset_name], dict)
        ):
            catalog_for_dataset = feature_catalog[dataset_name]
        if catalog_for_dataset is None:
            catalog_for_dataset = {}
        if not isinstance(catalog_for_dataset, dict):
            raise TypeError("feature_catalog must resolve to a dictionary.")
        verified_drop_metadata = {
            column: metadata
            for column, metadata in catalog_for_dataset.items()
            if isinstance(metadata, dict)
            and metadata.get("verified_drop_reason") in {
                "explicit_leakage", "verified_identifier"
            }
        }
        host_drop, host_drop_evidence = self._build_host_cleaning_audit(
            X_for_rf,
            verified_drop_metadata,
        )

        prompt_1 = build_stage1_user_prompt(
            dataset_context=dataset_context,
            summary_str=summary_str,
            numcorr_str=numcorr_str,
            catcorr_str=catcorr_str,
            target_assoc_str=fts_str,
            nzv_note=nzv_note,
            label_col_name=label_col_name,
            class_distribution=class_distribution,
            n_rows=n_rows,
            train_counts=train_counts,
            categorical_feature_cols=categorical_feature_cols,
            categorical_encoding_assignments=categorical_encoding_assignments,
            null_handling_assignments=null_handling_assignments,
            verified_drop_metadata=verified_drop_metadata,
        )
        # Multi-agent execution. Each stage runs as an ISOLATED agent
        # (`reset=True` starts a fresh conversation with its own role/system
        # prompt), and the explicit output of each stage is threaded into the
        # input of the next — this is what makes it a true agent chain rather
        # than one long single-agent conversation.
        # ------------------------------------------------------------------

        # --- Stage 1: Data Analyst -> preprocessing PLAN (JSON) ---
        # temperature=0: deterministic output for consistent plan generation
        # policies bounded; one self-correcting retry receives any validation error.
        all_feature_cols = numeric_feature_cols + categorical_feature_cols
        validated_decision = {}
        response_1 = self._call_validated(
            prompt_1,
            analyst_system_prompt,
            lambda reply: self._validate_stage1_decision(
                reply, all_feature_cols, train_counts, X_for_rf,
                verified_drop_metadata, validated_decision,
            ),
            temperature=0,
            label="Stage 1 (plan JSON)",
            base_seed=generation_seed,
        )

        decision_dict = validated_decision
        plan_dict = self._build_canonical_plan(
            decision_dict,
            train_counts,
            n_rows,
            categorical_encoding_assignments,
            null_handling_assignments,
            host_drop,
            host_drop_evidence,
        )

        clean_json = json.dumps(
            plan_dict,
            ensure_ascii=False,
            indent=2,
        )
        plan_ok, plan_error = self._validate_plan_json(
            clean_json,
            all_feature_cols,
            categorical_feature_cols,
            train_counts,
            n_rows,
            categorical_cardinalities,
            null_handling_assignments,
        )
        if not plan_ok:
            raise RuntimeError(
                "Host-generated canonical PLAN failed its internal contract: "
                f"{plan_error}"
            )

        # Stage 2 must copy these audited constants, not reconstruct them from prose.
        # Python repr is used deliberately so None/True/False remain valid module literals.
        approved_resampling_candidates = {
            item["candidate_id"]: item
            for item in plan_dict["resampling_candidates"]
        }
        approved_candidates_literal = repr(approved_resampling_candidates)
        approved_selection_literal = repr(plan_dict["resampling_selection"])
        approved_reference_rows_literal = repr(plan_dict["train_reference_rows"])
        approved_reference_counts_literal = repr(
            plan_dict["train_reference_class_counts"]
        )
        approved_encoding_policy_literal = repr(plan_dict["categorical_encoding_policy"])
        approved_scaling_policy_literal = repr(plan_dict["scaling_policy"])
        approved_log_policy_literal = repr(plan_dict["log_transform_policy"])
        approved_resampling_family = next(
            (
                item["imbalance_handling"]
                for item in plan_dict["resampling_candidates"]
                if item["imbalance_handling"] != "None"
            ),
            "None",
        )
        stage2_handoff = build_stage2_handoff(plan_dict)
        prompt_2 = build_stage2_user_prompt(
            dataset_context=dataset_context,
            stage2_handoff=stage2_handoff,
            approved_candidates_literal=approved_candidates_literal,
            approved_selection_literal=approved_selection_literal,
            approved_reference_rows_literal=approved_reference_rows_literal,
            approved_reference_counts_literal=approved_reference_counts_literal,
            approved_encoding_policy_literal=approved_encoding_policy_literal,
            approved_scaling_policy_literal=approved_scaling_policy_literal,
            approved_log_policy_literal=approved_log_policy_literal,
            approved_resampling_family=approved_resampling_family,
        )
        # Keep the canonical final PLAN for history/logging. Stage 2 receives only
        # the compact handoff plus authoritative literals built above.
        response_1 = clean_json
        print(response_1)

        dropped_features = plan_dict.get("drop", [])

        # Exclude dropped features from both feature lists.
        filtered_numeric_cols = [c for c in numeric_feature_cols if c not in dropped_features]
        filtered_categorical_cols = [c for c in categorical_feature_cols if c not in dropped_features]

        # Optional per-dataset semantics. Existing callers remain valid; undocumented features
        stage3_original_cols = filtered_numeric_cols + filtered_categorical_cols
        # stay eligible for RF/MI and are never classified by a built-in keyword/root list.
        if not stage3_original_cols:
            raise ValueError(
                "Stage 3 adaptive RF+MI selection requires at least one post-drop "
                "feature."
            )
        feature_catalog_str = self._build_feature_catalog(
            X_for_rf[[col for col in stage3_original_cols if col in X_for_rf.columns]],
            catalog_for_dataset,
        )

        prompt_3 = build_stage3_user_prompt(
            dataset_context=dataset_context,
            n_rows=n_rows,
            classification_type=classification_type,
            filtered_numeric_cols=filtered_numeric_cols,
            filtered_categorical_cols=filtered_categorical_cols,
            feature_catalog_str=feature_catalog_str,
        )
        # --- Stage 2: Code Engineer -> Python cleaning function implementing the PLAN ---
        # temperature=0: deterministic output for code generation — strictly follows instructions.
        # Validated for syntax, forbidden TEST-prior use, rare-class preservation, and categorical coverage.
        # Train-derived samples only: include every TRAIN class (including ultra-rare classes)
        # without ever reading held-out TEST labels. The validator first isolates common
        # preprocessing with candidate 'none', then safely executes every approved candidate
        # against its proportionally scaled compact-fixture budget.
        contract_indices = train_df.groupby(label_col_name, observed=True).head(8).index
        contract_train = X_for_rf.loc[contract_indices].copy()
        contract_test = X_for_rf.tail(min(24, len(X_for_rf))).copy()
        contract_y = train_df.loc[contract_indices, label_col_name].copy()
        response_2 = self._call_validated(
            prompt_2,
            engineer_system_prompt,
            lambda reply: self._validate_preprocessor_code(
                reply, contract_train, contract_test, contract_y, plan_dict
            ),
            temperature=0,
            label="Stage 2 (cleaning code)",
            max_retries=2,
            rejection_log_dir=_REJECTED_DIR / generation_run_id / "stage2",
            base_seed=generation_seed,
        )
        print(response_2)

        # --- Stage 3: Feature-Reduction Specialist -> candidate-agnostic adaptive-K reducer code ---
        # Stage 3 receives only the post-drop schema/catalog. The validation host
        # executes its reducer once on candidate 'none' and locks that feature list.
        # temperature=0: deterministic code generation. Validated for type-aware multiclass scoring,
        # label-name invariance, and absence of regex/name-based hard feature rules.
        response_3 = self._call_validated(
            prompt_3,
            feature_system_prompt,
            lambda reply: self._validate_reducer_code(
                reply, filtered_numeric_cols, filtered_categorical_cols
            ),
            temperature=0,
            label="Stage 3 (reducer code)",
            max_retries=2,
            rejection_log_dir=_REJECTED_DIR / generation_run_id / "stage3",
            base_seed=generation_seed,
        )
        print(response_3)

        # Assemble and syntax-check both accepted stages before publishing anything
        # into generated/. A Stage-3 failure therefore cannot leave a partial artifact.
        module_code, reduction_kind = self._compose_generated_module(
            response_2,
            response_3,
        )
        print(f"[LLM] REDUCTION_KIND = {reduction_kind}")
        saved_file, timestamp = self._write_generated_file(
            module_code, self.model_name, classification_type
        )
        print(f"[LLM] Generated code saved → generated/{saved_file}")

        # Record history now that the chosen reduction kind is known (cache key = "llm").
        self._record_history(timestamp, saved_file, self.model_name, dataset_name, classification_type, reduction_kind)

        # Save the full responses to txt file
        with open(f"text_files/feature_{reduction_kind}_llm_suggestions.txt", "w", encoding='utf-8') as file:
            file.write(response_1)
            file.write("\n\n\n")
            file.write(response_2)
            file.write("\n\n\n")
            file.write(response_3)

        # Print per-generation usage summary
        usage = self.api_caller.usage_summary()
        print(
            f"[LLM usage] calls={usage['calls']} | "
            f"prompt={usage['prompt_tokens']} tok, completion={usage['completion_tokens']} tok, "
            f"total={usage['total_tokens']} tok | "
            f"api_latency={usage['api_latency_seconds']:.2f}s | "
            f"cost=${usage['cost_usd']:.6f}"
        )

        # Reset conversation for next call
        self.api_caller.messages = []

        return saved_file
