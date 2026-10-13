"""
Feature selection data preprocessing utilities
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder



class FeatureSelectionDataPreprocessor:
    def __init__(self, dataset_name: str):
        pass
        self.dataset_name = dataset_name

    @staticmethod
    def avg_correlation_indexes(train_set):
        correlation_matrix = train_set.corr(method='pearson').abs().fillna(0.0)

        avg_correlation = correlation_matrix.mean(axis=1).sort_values(ascending=False)
        avg_correlation = avg_correlation.reset_index()
        avg_correlation.columns = ['feature', 'average_correlation']
        return avg_correlation

    @staticmethod
    def feature_selection_preprocessing(avg_correlation_df, n_features):
        avg_correlation_df = avg_correlation_df.sort_values(by='average_correlation', ascending=False)
        avg_correlation_df = avg_correlation_df.iloc[:n_features, :]
        return avg_correlation_df['feature'].tolist()


class FeatureSelectionDataPreprocessorV2:
    def __init__(self, dataset_name: str):
        pass
        self.dataset_name = dataset_name

    @staticmethod
    def _numeric_training_matrix(X: pd.DataFrame) -> pd.DataFrame:
        """Return a finite numeric matrix fitted from TRAIN columns only."""
        X_numeric = X.apply(pd.to_numeric, errors="coerce")
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        return X_numeric.fillna(X_numeric.median()).fillna(0.0)

    @staticmethod
    def _encode_target(y: pd.Series) -> np.ndarray:
        """Encode arbitrary class labels deterministically for RF and MI."""
        y_series = pd.Series(y).reset_index(drop=True)
        if y_series.isna().any():
            raise ValueError("y must not contain missing values")
        return LabelEncoder().fit_transform(y_series)

    @staticmethod
    def _discrete_feature_mask(X: pd.DataFrame) -> np.ndarray:
        """Mirror Stage 3's type-aware discrete/continuous MI routing."""
        n_train = len(X)
        int_like_cap = min(50, max(2, int(np.floor(np.sqrt(max(1, n_train))))))
        mask = []

        for col in X.columns:
            values = X[col].to_numpy(dtype=float, copy=False)
            unique_values = np.unique(values)
            is_binary01 = set(unique_values.tolist()).issubset({0.0, 1.0})
            integer_like = np.all(
                np.isclose(values, np.round(values), rtol=0.0, atol=1e-12)
            )
            mask.append(bool(is_binary01 or (integer_like and len(unique_values) <= int_like_cap)))

        return np.asarray(mask, dtype=bool)

    @staticmethod
    def _percentile_scores(values: np.ndarray) -> np.ndarray:
        """Convert one score family to deterministic Stage-3 percentile ranks."""
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return values
        if len(values) == 1:
            return np.ones(1, dtype=float)

        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float) / float(len(values) - 1)
        return ranks

    # Rank features by target relevance. RF+MI mirrors Stage 3 and is the default.
    @staticmethod
    def compute_feature_target_correlation(
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "rf_mi",
        random_state: int = 42,
    ) -> pd.DataFrame:
        if method not in {"rf_mi", "hybrid", "mutual_info", "pearson", "cosine"}:
            raise ValueError(
                "method must be 'rf_mi'/'hybrid', 'mutual_info', 'pearson', or 'cosine'"
            )

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        if X.empty:
            return pd.DataFrame(columns=["feature", "average_correlation"])

        if method in {"rf_mi", "hybrid", "mutual_info"}:
            X_numeric = FeatureSelectionDataPreprocessorV2._numeric_training_matrix(X)
            y_encoded = FeatureSelectionDataPreprocessorV2._encode_target(y)

            if len(np.unique(y_encoded)) < 2:
                mi_scores = np.zeros(X_numeric.shape[1], dtype=float)
            else:
                mi_scores = mutual_info_classif(
                    X_numeric,
                    y_encoded,
                    discrete_features=FeatureSelectionDataPreprocessorV2._discrete_feature_mask(
                        X_numeric
                    ),
                    random_state=random_state,
                )
                mi_scores = np.nan_to_num(
                    mi_scores, nan=0.0, posinf=0.0, neginf=0.0
                )

            if method == "mutual_info":
                scores = mi_scores
            else:
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=None,
                )
                rf.fit(X_numeric, y_encoded)
                rf_scores = np.nan_to_num(
                    rf.feature_importances_, nan=0.0, posinf=0.0, neginf=0.0
                )
                rf_percentiles = FeatureSelectionDataPreprocessorV2._percentile_scores(
                    rf_scores
                )
                mi_percentiles = FeatureSelectionDataPreprocessorV2._percentile_scores(
                    mi_scores
                )
                scores = 0.60 * rf_percentiles + 0.40 * mi_percentiles
        else:
            y_vals = pd.Series(y).to_numpy(dtype=float)
            scores = []

            if method == "pearson":
                for col in X.columns:
                    s = X[col].values.astype(float)
                    corr = pd.Series(s).corr(pd.Series(y_vals))
                    scores.append(float(abs(corr)) if corr is not None else 0.0)
            else:  # cosine
                y_norm = np.linalg.norm(y_vals)
                if y_norm == 0:
                    y_norm = 1e-12
                for col in X.columns:
                    v = X[col].values.astype(float)
                    v_norm = np.linalg.norm(v)
                    if v_norm == 0:
                        score = 0.0
                    else:
                        score = float(abs(np.dot(v, y_vals) / (v_norm * y_norm)))
                    scores.append(score)

        # Keep the existing score-column name for backward compatibility.
        rank_data = {
            "feature": X.columns,
            "average_correlation": scores,
        }
        if method in {"rf_mi", "hybrid"}:
            rank_data.update({
                "rf_importance": rf_scores,
                "mutual_info": mi_scores,
                "rf_percentile": rf_percentiles,
                "mi_percentile": mi_percentiles,
            })

        rank_df = pd.DataFrame(rank_data)
        rank_df = rank_df.sort_values(
            ["average_correlation", "feature"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        return rank_df

    # Feature-feature correlation matrix using Pearson or Cosine
    @staticmethod
    def _feature_feature_correlation(
        X: pd.DataFrame,
        method: str = "pearson"
    ) -> pd.DataFrame:
        X_numeric = FeatureSelectionDataPreprocessorV2._numeric_training_matrix(X)
        if method == "pearson":
            C = X_numeric.corr(method='pearson').abs()
            return C.fillna(0.0)
        elif method == "cosine":
            # Normalize each column into a unit vector, then perform matrix multiplication
            V = X_numeric.astype(float).values
            # Replace NaN with 0 before normalization
            V = np.nan_to_num(V, nan=0.0)
            norms = np.linalg.norm(V, axis=0)
            norms[norms == 0] = 1e-12
            Vn = V / norms
            C = np.abs(np.dot(Vn.T, Vn))  # absolute cosine similarity
            return pd.DataFrame(C, index=X.columns, columns=X.columns)
        else:
            raise ValueError("method must be 'pearson' or 'cosine'")

    # Filter multicollinearity after relevance ranking (RF+MI by default).
    @staticmethod
    def feature_selection_preprocessing(
        avg_correlation_df: pd.DataFrame,
        n_features: int,
        train_set: Optional[pd.DataFrame] = None,
        threshold: float = 0.95,
        redundancy_method: str = "pearson"
    ) -> List[str]:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if not 0.95 <= float(threshold) <= 0.99:
            raise ValueError("threshold must be between 0.95 and 0.99 inclusive")
        if not {"feature", "average_correlation"}.issubset(avg_correlation_df.columns):
            raise ValueError(
                "avg_correlation_df must contain 'feature' and 'average_correlation' columns"
            )

        ranked = (
            avg_correlation_df
            .sort_values(
                ["average_correlation", "feature"],
                ascending=[False, True],
                kind="mergesort",
            )
            .drop_duplicates(subset="feature", keep="first")
            .reset_index(drop=True)
        )
        ranked_features = ranked["feature"].tolist()
        if n_features > len(ranked_features):
            raise ValueError(
                f"n_features={n_features} exceeds ranked features={len(ranked_features)}"
            )

        # Fallback case: no data available to filter for multicollinearity
        if train_set is None:
            top = ranked.iloc[:n_features, :]
            return top['feature'].tolist()

        missing_features = [feat for feat in ranked_features if feat not in train_set.columns]
        if missing_features:
            raise KeyError(f"Ranked features missing from train_set: {missing_features}")

        # Calculate feature-feature correlation to filter for redundancy
        X = train_set[ranked_features]  # ensure column order matches the ranking
        ff_corr = FeatureSelectionDataPreprocessorV2._feature_feature_correlation(X, method=redundancy_method)

        selected: List[str] = []
        current_threshold = round(float(threshold), 2)
        while len(selected) < n_features:
            for feat in ranked_features:
                if feat in selected:
                    continue
                if selected:
                    max_corr = float(ff_corr.loc[feat, selected].max())
                    if max_corr > current_threshold:
                        continue
                selected.append(feat)
                if len(selected) == n_features:
                    break

            if len(selected) == n_features or current_threshold >= 0.99:
                break
            current_threshold = round(min(0.99, current_threshold + 0.01), 2)

        # This legacy helper fills K columns even if every remaining pair exceeds 0.99.
        if len(selected) < n_features:
            for feat in ranked_features:
                if feat not in selected:
                    selected.append(feat)
                    if len(selected) == n_features:
                        break

        return selected[:n_features]
