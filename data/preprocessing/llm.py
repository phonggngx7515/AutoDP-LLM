"""Apply dataset-specific cleanup and generated multiclass preprocessing."""

from typing import Tuple

import numpy as np
import pandas as pd

from .prior import (
    CIC_IDS_2017_PriorPreprocessor,
    NSL_KDD_PriorPreprocessor,
    UNSW_NB15_PriorPreprocessor,
)


class LLMDataPreprocessor:
    def __init__(self, dataset: str, llm: object):
        preprocessors = {
            "UNSW_NB15": UNSW_NB15_PriorPreprocessor,
            "CIC_IDS_2017": CIC_IDS_2017_PriorPreprocessor,
            "NSL_KDD": NSL_KDD_PriorPreprocessor,
        }
        if dataset not in preprocessors:
            raise ValueError(f"Unsupported dataset name: {dataset}")
        self.prior_preprocessor = preprocessors[dataset]
        self.llm = llm

    def multi_class_LLM_preprocessing(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "attack_cat",
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Return transformed TRAIN/TEST features and their separate multiclass labels."""
        train_data = self.prior_preprocessor.multi_class_prior_preprocessing(train_df.copy())
        test_data = self.prior_preprocessor.multi_class_prior_preprocessing(test_df.copy())
        train_label = train_data.pop(target_col)
        test_label = test_data.pop(target_col)

        if "service" in train_data.columns:
            train_data["service"] = train_data["service"].replace("-", "other")
        if "service" in test_data.columns:
            test_data["service"] = test_data["service"].replace("-", np.nan)

        # The generated function may resample TRAIN and must return aligned TRAIN labels.
        # Feature selection is applied separately through the artifact's reduce_features().
        train_processed, train_label, test_processed = self.llm.preprocess_data(
            train_data, test_data, train_label
        )
        return train_processed, train_label, test_processed, test_label
