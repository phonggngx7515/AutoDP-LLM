"""
Base class for all models.
Provides a common interface for training and evaluation.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# from sklearn.svm import SVC

# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.callbacks import EarlyStopping


class BinaryReturnMetrics:

    @staticmethod
    def return_metrics(labels, preds):
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }

class MultiClassReturnMetrics:

    @staticmethod
    def return_metrics(labels, preds):
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }

class ResultReport:

    @staticmethod
    def show_fig(anomaly_scores, threshold):
        normal_scores = anomaly_scores[anomaly_scores <= threshold]
        abnormal_scores = anomaly_scores[anomaly_scores > threshold]

        plt.figure(figsize=(10, 6))
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='skyblue')
        plt.hist(abnormal_scores, bins=50, alpha=0.7, label='Attack', color='sandybrown')

        plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')

        plt.xlabel('Reconstruction Error')
        plt.ylabel('Number of Samples')
        plt.legend()

        plt.show()

    @staticmethod
    def show_report(labels, preds):
        print("\nClassification Report:\n")
        report = classification_report(
            labels,
            preds,
            zero_division=0,
            output_dict=True,
        )
        report.pop("macro avg", None)
        print(pd.DataFrame(report).transpose().to_string())

    @staticmethod
    def results_df(model_results: dict) -> pd.DataFrame:
        """Build a results DataFrame from a dict of {model_name: metrics_dict}.

        metrics_dict may contain: 'Accuracy', 'Precision', 'Recall', 'F1', 'Time (s)'.
        """
        rows = []
        for name, m in model_results.items():
            rows.append({
                "Model": name,
                "Accuracy": m.get("Accuracy"),
                "Precision": m.get("Precision"),
                "Recall": m.get("Recall"),
                "F1 Score": m.get("F1"),
                "Time (s)": m.get("Time (s)"),
            })
        return pd.DataFrame(rows)


class BaseModel(BaseEstimator, TransformerMixin, ResultReport):
    def __init__(self, type_of_classification=None):
        if type_of_classification == 'binary':
            self.classification_type = BinaryReturnMetrics
        elif type_of_classification == 'multi_class':
            self.classification_type = MultiClassReturnMetrics
        else:
            raise ValueError("type_of_classification must be 'binary' or 'multi_class'")

    def _train_and_eval(self, name, model, train_data, train_label, test_data, test_label):
        """Fit, predict, time and evaluate any sklearn-compatible model."""
        print(f"{name} - METRICS:")
        t0 = time.perf_counter()
        
        # Check if train_label has string/object categories
        is_string_target = False
        if isinstance(train_label, pd.Series):
            is_string_target = train_label.dtype == "object" or str(train_label.dtype).startswith("string")
        elif isinstance(train_label, np.ndarray):
            is_string_target = train_label.dtype.kind in {'U', 'S', 'O'}
            
        if is_string_target:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            # Fit only on training labels so that class integers are exactly [0, num_classes_train - 1]
            le.fit(train_label)
            
            train_label_fit = le.transform(train_label)
            
            # Map unseen test labels defensively to -1
            test_label_fit = np.array([le.transform([x])[0] if x in le.classes_ else -1 for x in test_label])
            
            model.fit(train_data, train_label_fit)
            predictions_fit = model.predict(test_data)
            elapsed = time.perf_counter() - t0
            
            # Compute metrics using the encoded labels
            metrics = self.classification_type.return_metrics(test_label_fit, predictions_fit)
            metrics["Time (s)"] = elapsed
            print(f"  Elapsed: {elapsed:.2f}s")
            
            # Show report using original string labels
            predictions = le.inverse_transform(predictions_fit)
            self.show_report(test_label, predictions)
        else:
            model.fit(train_data, train_label)
            predictions = model.predict(test_data)
            elapsed = time.perf_counter() - t0
            metrics = self.classification_type.return_metrics(test_label, predictions)
            metrics["Time (s)"] = elapsed
            print(f"  Elapsed: {elapsed:.2f}s")
            self.show_report(test_label, predictions)
            
        return metrics

    def DecisionTreeModel(self, train_data, train_label, test_data, test_label):
        return self._train_and_eval(
            "DECISION TREE",
            DecisionTreeClassifier(random_state=42),
            train_data, train_label, test_data, test_label,
        )

    def RandomForestModel(self, train_data, train_label, test_data, test_label):
        return self._train_and_eval(
            "RANDOM FOREST",
            RandomForestClassifier(random_state=42),
            train_data, train_label, test_data, test_label,
        )

    def KNNModel(self, train_data, train_label, test_data, test_label):
        return self._train_and_eval(
            "K-NEAREST NEIGHBORS",
            KNeighborsClassifier(n_neighbors=5),
            train_data, train_label, test_data, test_label,
        )

    def MLPModel(self, train_data, train_label, test_data, test_label):
        return self._train_and_eval(
            "MLP",
            MLPClassifier(random_state=42),
            train_data, train_label, test_data, test_label,
        )

    def BernoulliNBModel(self, train_data, train_label, test_data, test_label):
        return self._train_and_eval(
            "BERNOULLI NAIVE BAYES",
            BernoulliNB(),
            train_data, train_label, test_data, test_label,
        )

    def XGBoostModel(self, train_data, train_label, test_data, test_label):
        return self._train_and_eval(
            "XGBOOST",
            XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
            train_data, train_label, test_data, test_label,
        )

    def LGBMModel(self, train_data, train_label, test_data, test_label):
        return self._train_and_eval(
            "LIGHTGBM",
            LGBMClassifier(random_state=42, verbosity=-1),
            train_data, train_label, test_data, test_label,
        )

    # def SVMModel(self, train_data, train_label, test_data, test_label):
    #     return self._train_and_eval(
    #         "SVM",
    #         SVC(random_state=42, kernel="rbf"),
    #         train_data, train_label, test_data, test_label,
    #     )

    def CatBoostModel(self, train_data, train_label, test_data, test_label):
        return self._train_and_eval(
            "CATBOOST",
            CatBoostClassifier(random_seed=42, verbose=0),
            train_data, train_label, test_data, test_label,
        )

    def AutoEncoderModel(self, train_data, train_label, test_data, test_label):
        # Architecture
        model = Sequential([
            Input(shape=(train_data.shape[1],)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(train_data.shape[1])
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

        # Isolate normal train data
        normal_train_data = train_data[train_label == 0]

        # Train model (on normal data only)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        print("TRAINING MODEL:")
        t0 = time.perf_counter()
        record = model.fit(normal_train_data, normal_train_data,
                        epochs=100,
                        batch_size=128,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=[early_stop],)
        print()

        # Learning Curves
        print("LEARNING CURVES:")
        history_df = pd.DataFrame(record.history)
        history_df.loc[:, ['loss', 'val_loss']].plot(title='MSE')
        plt.show()
        print()

        # Anomaly Scores / Inference on complete and new data
        reconstructed_data = model.predict(test_data)
        anomaly_scores = np.linalg.norm((test_data - reconstructed_data), axis=1)

        # Threshold
        print("CALCULATING OPTIMAL THRESHOLD:")
        fpr, tpr, thresholds = roc_curve(test_label, anomaly_scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
        print(f"Optimal threshold: {threshold:.4f}")
        print()

        # Predictions
        anomaly_preds = (anomaly_scores >= threshold).astype(int)
        elapsed = time.perf_counter() - t0

        # Metrics
        print("AUTO ENCODER - METRICS:")
        metrics = self.classification_type.return_metrics(test_label, anomaly_preds)
        metrics["Time (s)"] = elapsed
        print(f"  Elapsed: {elapsed:.2f}s")
        print()

        # Classification Report
        self.show_report(test_label, anomaly_preds)
        print()

        # Reconstruction error distributions;
        print("RECONSTRUCTION ERROR DISTRIBUTIONS:")
        self.show_fig(anomaly_scores, threshold)

        return metrics

    def StackingModel(self, train_data, train_label, test_data, test_label):
        base_models = [
            ('dt', DecisionTreeClassifier(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('xgb', XGBClassifier(random_state=42))
        ]
        meta_model = MLPClassifier(random_state=42)
        return self._train_and_eval(
            "STACKING",
            StackingClassifier(estimators=base_models, final_estimator=meta_model),
            train_data, train_label, test_data, test_label,
        )

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

    def evaluate_all_models(self, train_data, train_label, test_data, test_label):

        # Evaluate all models
        model_results = {
            "Decision Tree": self.DecisionTreeModel(train_data, train_label, test_data, test_label),
            "Random Forest": self.RandomForestModel(train_data, train_label, test_data, test_label),
            "KNN":           self.KNNModel(train_data, train_label, test_data, test_label),
            "MLP":           self.MLPModel(train_data, train_label, test_data, test_label),
            "Bernoulli NB":  self.BernoulliNBModel(train_data, train_label, test_data, test_label),
            "XGBoost":       self.XGBoostModel(train_data, train_label, test_data, test_label),
            "LightGBM":      self.LGBMModel(train_data, train_label, test_data, test_label),
            "CatBoost":      self.CatBoostModel(train_data, train_label, test_data, test_label),
            # "SVM":           self.SVMModel(train_data, train_label, test_data, test_label),
        }

        results_df = self.results_df(model_results)
        total_time = sum((m.get("Time (s)") or 0.0) for m in model_results.values())
        print(f"\nTotal model evaluation time: {total_time:.2f}s")
        print("RESULTS:")
        return results_df
