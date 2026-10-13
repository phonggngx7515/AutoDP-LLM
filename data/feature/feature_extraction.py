"""
Feature extraction data preprocessing utilities
"""

import pandas as pd
from sklearn.decomposition import PCA


class FeatureExtractionDataPreprocessor:
    def __init__(self, dataset_name: str):
        pass
        self.dataset_name = dataset_name

    @staticmethod
    def feature_extraction(train_set, test_set, n_components):
        pca = PCA(n_components=n_components)
        train = pd.DataFrame(pca.fit_transform(train_set))
        test = pd.DataFrame(pca.transform(test_set))
        return train, test
