"""Shared multiclass classifier factories and weighted evaluation metrics."""

import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class BaseModel:
    """Evaluate all supported classifiers with the same multiclass metric contract."""

    @staticmethod
    def classifier_factories():
        """Return fresh final classifiers in the canonical reporting order."""
        return {
            "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
            "Random Forest": lambda: RandomForestClassifier(random_state=42),
            "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
            "MLP": lambda: MLPClassifier(random_state=42),
            "Bernoulli NB": lambda: BernoulliNB(),
            "XGBoost": lambda: XGBClassifier(
                random_state=42, eval_metric="logloss", verbosity=0
            ),
            "LightGBM": lambda: LGBMClassifier(random_state=42, verbosity=-1),
            "CatBoost": lambda: CatBoostClassifier(random_seed=42, verbose=0),
        }

    @staticmethod
    def evaluate_classifier(
        classifier_name,
        train_data,
        train_label,
        evaluation_data,
        evaluation_label,
    ):
        """Fit one final classifier and return compact deterministic metrics.

        This is shared by validation and the single held-out TEST evaluation so
        candidate selection cannot silently use different model hyperparameters.
        """
        factories = BaseModel.classifier_factories()
        if classifier_name not in factories:
            raise ValueError(f"Unknown classifier: {classifier_name!r}")

        encoder = LabelEncoder()
        train_series = pd.Series(train_label).reset_index(drop=True)
        evaluation_series = pd.Series(evaluation_label).reset_index(drop=True)
        encoded_train = encoder.fit_transform(train_series)
        try:
            encoded_evaluation = encoder.transform(evaluation_series)
        except ValueError as exc:
            raise ValueError(
                "Evaluation labels contain a class absent from the fitted TRAIN partition."
            ) from exc

        model = factories[classifier_name]()
        started = time.perf_counter()
        model.fit(train_data, encoded_train)
        predictions = np.asarray(model.predict(evaluation_data), dtype=int)
        elapsed = time.perf_counter() - started
        return {
            "accuracy": float(accuracy_score(encoded_evaluation, predictions)),
            "weighted_precision": float(precision_score(
                encoded_evaluation,
                predictions,
                average="weighted",
                zero_division=0,
            )),
            "weighted_recall": float(recall_score(
                encoded_evaluation,
                predictions,
                average="weighted",
                zero_division=0,
            )),
            "weighted_f1": float(f1_score(
                encoded_evaluation,
                predictions,
                average="weighted",
                zero_division=0,
            )),
            "runtime_seconds": float(elapsed),
        }
