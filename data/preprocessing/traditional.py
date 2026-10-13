"""
Traditional data preprocessing utilities
"""

import pandas as pd
import numpy as np
from numpy import hstack
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from .prior import UNSW_NB15_PriorPreprocessor, CIC_IDS_2017_PriorPreprocessor, NSL_KDD_PriorPreprocessor, BotNetIoT_L01_PriorPreprocessor

class TraditionalDataPreprocessor:
    def __init__(self, dataset_name: str):
        if dataset_name == "UNSW_NB15":
            self.prior_preprocessor = UNSW_NB15_PriorPreprocessor
        elif dataset_name == "CIC_IDS_2017":
            self.prior_preprocessor = CIC_IDS_2017_PriorPreprocessor
        elif dataset_name == "NSL_KDD":
            self.prior_preprocessor = NSL_KDD_PriorPreprocessor
        elif dataset_name == "BotNetIoT_L01":
            self.prior_preprocessor = BotNetIoT_L01_PriorPreprocessor
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")
        
    def binary_traditional_preprocessing(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = 'label') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Preprocess train and test sets for binary classification.

        Args:
            train_df (pd.DataFrame): Training data with 'label' column as labels.
            test_df (pd.DataFrame): Test data with 'label' column as labels.

        Returns:
            Tuple: (preprocessed_train_set, train_label, preprocessed_test_set, test_label)
        """
        # --- TRAIN DATA ---
        train_set = train_df.copy()
        train_set = self.prior_preprocessor.binary_prior_preprocessing(train_set)
        if 'service' in train_set.columns:
            train_set['service'] = train_set['service'].replace(np.nan, 'other')
        train_label = train_set.pop(target_col)

        numerical_cols = train_set.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = train_set.select_dtypes(include=['object']).columns.tolist()

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        scaler = MinMaxScaler()

        train_cat = encoder.fit_transform(train_set[categorical_cols])
        train_num = scaler.fit_transform(train_set[numerical_cols])

        preprocessed_train_set = pd.DataFrame(
            hstack([train_num, train_cat]),
            columns=numerical_cols + list(encoder.get_feature_names_out(categorical_cols)),
            index=train_set.index
        )

        # --- TEST DATA ---
        test_set = test_df.copy()
        test_set = self.prior_preprocessor.binary_prior_preprocessing(test_set)
        if 'service' in test_set.columns:
            test_set['service'] = test_set['service'].replace(np.nan, 'other')
        test_label = test_set.pop(target_col)

        test_cat = encoder.transform(test_set[categorical_cols])
        test_num = scaler.transform(test_set[numerical_cols])

        preprocessed_test_set = pd.DataFrame(
            hstack([test_num, test_cat]),
            columns=numerical_cols + list(encoder.get_feature_names_out(categorical_cols)),
            index=test_set.index
        )

        return preprocessed_train_set, train_label, preprocessed_test_set, test_label
    
    def multi_class_traditional_preprocessing(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = 'attack_cat') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Preprocess train and test sets for multi-class classification.

        Args:
            train_df (pd.DataFrame): Training data with 'attack_cat' column as labels.
            test_df (pd.DataFrame): Test data with 'attack_cat' column as labels.

        Returns:
            Tuple: (preprocessed_train_set, train_label, preprocessed_test_set, test_label)
        """
        # --- TRAIN DATA ---
        train_set = train_df.copy()
        train_set = self.prior_preprocessor.multi_class_prior_preprocessing(train_set)
        if 'service' in train_set.columns:
            train_set['service'] = train_set['service'].replace(np.nan, 'other')
        train_label = train_set.pop(target_col)

        numerical_cols = train_set.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = train_set.select_dtypes(include=['object']).columns.tolist()

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        scaler = MinMaxScaler()
        train_cat = encoder.fit_transform(train_set[categorical_cols])
        train_num = scaler.fit_transform(train_set[numerical_cols])

        preprocessed_train_set = pd.DataFrame(
            hstack([train_num, train_cat]),
            columns=numerical_cols + list(encoder.get_feature_names_out(categorical_cols)),
            index=train_set.index
        )

        # --- TEST DATA ---
        test_set = test_df.copy()
        test_set = self.prior_preprocessor.multi_class_prior_preprocessing(test_set)
        if 'service' in test_set.columns:
            test_set['service'] = test_set['service'].replace(np.nan, 'other')
        test_label = test_set.pop(target_col)

        test_cat = encoder.transform(test_set[categorical_cols])
        test_num = scaler.transform(test_set[numerical_cols])

        preprocessed_test_set = pd.DataFrame(
            hstack([test_num, test_cat]),
            columns=numerical_cols + list(encoder.get_feature_names_out(categorical_cols)),
            index=test_set.index
        )

        return preprocessed_train_set, train_label, preprocessed_test_set, test_label
    
class TraditionalDataPreprocessorV2:
    def __init__(self, dataset_name: str):
        if dataset_name == "UNSW_NB15":
            self.prior_preprocessor = UNSW_NB15_PriorPreprocessor
        elif dataset_name == "CIC_IDS_2017":
            self.prior_preprocessor = CIC_IDS_2017_PriorPreprocessor
        elif dataset_name == "NSL_KDD":
            self.prior_preprocessor = NSL_KDD_PriorPreprocessor
        else:
            raise ValueError("Unsupported dataset name. Choose 'UNSW_NB15', 'CIC_IDS_2017', or 'NSL_KDD'.")

    def preprocess_unsw_nb15(self, df, df_name):
        """
        Cleans, sanitizes, and prepares a single UNSW-NB15 dataframe.
        """
        print(f"--- Starting Preprocessing for {df_name} ---")
        print(f"Initial shape: {df.shape}")

        # --- 1. Column Name and Type Correction ---
        # Drop the 'id' column as it's just an identifier
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            print("Dropped 'id' column.")
        
        # The 'attack_cat' column is a specific label category, not a feature for binary classification
        # We will drop it. The 'label' column will be our target.
        if 'attack_cat' in df.columns:
            df = df.drop('attack_cat', axis=1)
            print("Dropped 'attack_cat' column.")

        # Clean column names by stripping leading/trailing whitespace
        df.columns = df.columns.str.strip()

        # --- 2. Sanitize and Clean Data ---
        # First, let's identify our true categorical features. These should NOT be converted to numeric.
        categorical_features = ['proto', 'service', 'state']
        
        # Replace non-numeric or placeholder strings with NaN
        # A robust way to handle this is to convert to numeric, coercing errors
        potential_numeric_cols = df.select_dtypes(include=['object']).columns.difference(categorical_features)
        for col in potential_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Replace infinite values with NaN to handle them in one step
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop rows with any NaN values
        rows_before_na_drop = df.shape[0]
        df.dropna(inplace=True)
        print(f"Dropped {rows_before_na_drop - df.shape[0]} rows with NaN or Inf values.")

        # Exclude duplicate rows
        if (df_name == "Training Data"):
            rows_before_dup_drop = df.shape[0]
            df.drop_duplicates(inplace=True)
            print(f"Dropped {rows_before_dup_drop - df.shape[0]} duplicate rows.")
        
        # The 'label' column should be integer type
        df['label'] = df['label'].astype(int)

        print(f"Shape after cleaning: {df.shape}")
        return df

    def binary_traditional_preprocessing(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """
        Main function to run the entire preprocessing pipeline for the UNSW-NB15 dataset.
        """

        # --- Run Cleaning and Sanitizing ---
        df_train_clean = self.preprocess_unsw_nb15(df_train.copy(), "Training Data")
        df_test_clean = self.preprocess_unsw_nb15(df_test.copy(), "Testing Data")

        # Separate features (X) and the target (y)
        X_train = df_train_clean.drop('label', axis=1)
        y_train = df_train_clean['label']
        X_test = df_test_clean.drop('label', axis=1)
        y_test = df_test_clean['label']

        # --- 3. Data Encoding (Convert Categorical to Numeric) ---
        print("\n--- Encoding Categorical Features ---")
        # Using pandas get_dummies is a straightforward way to do one-hot encoding
        X_train_encoded = pd.get_dummies(X_train)
        X_test_encoded = pd.get_dummies(X_test)
        
        # Align columns after encoding. This is CRUCIAL!
        train_cols = X_train_encoded.columns
        test_cols = X_test_encoded.columns
        
        missing_in_test = set(train_cols) - set(test_cols)
        for c in missing_in_test:
            X_test_encoded[c] = 0
            
        missing_in_train = set(test_cols) - set(train_cols)
        for c in missing_in_train:
            X_train_encoded[c] = 0
            
        # Ensure the order of columns is the same
        X_test_encoded = X_test_encoded[X_train_encoded.columns]

        print(f"Training data shape after encoding: {X_train_encoded.shape}")
        print(f"Testing data shape after encoding: {X_test_encoded.shape}")

        # --- 4. Undersampling (on Training Data ONLY) ---
        print("\n--- Balancing Training Data with Undersampling ---")
        # Combine encoded features and labels to make sampling easier
        df_train_encoded = pd.concat([X_train_encoded, y_train], axis=1)
        
        # Get the counts for each class
        attack_count = df_train_encoded['label'].value_counts().get(1, 0)
        normal_count = df_train_encoded['label'].value_counts().get(0, 0)
        
        print(f"Before undersampling -> Normal (0): {normal_count}, Attack (1): {attack_count}")

        if attack_count > 0 and normal_count > attack_count:
            df_attack = df_train_encoded[df_train_encoded['label'] == 1]
            df_normal = df_train_encoded[df_train_encoded['label'] == 0]
            
            # Undersample the majority class (normal)
            df_normal_undersampled = df_normal.sample(n=attack_count, random_state=42)
            
            df_train_balanced = pd.concat([df_normal_undersampled, df_attack], axis=0)
        else:
            print("Skipping undersampling as attack class is not the minority.")
            df_train_balanced = df_train_encoded

        # Shuffle the dataset
        df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"After undersampling -> Normal (0): {df_train_balanced['label'].value_counts().get(0, 0)}, Attack (1): {df_train_balanced['label'].value_counts().get(1, 0)}")

        # Separate features and labels again
        X_train_balanced = df_train_balanced.drop('label', axis=1)
        y_train_balanced = df_train_balanced['label']

        # --- 5. Min-Max Scaling (Normalization) ---
        print("\n--- Normalizing Data with Min-Max Scaler ---")
        scaler = MinMaxScaler()
        
        scaler.fit(X_train_balanced)
        
        X_train_scaled = scaler.transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test_encoded)
        
        X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_balanced.columns)
        X_test_final = pd.DataFrame(X_test_scaled, columns=X_test_encoded.columns)

        print("\nPreprocessing complete! ✨ Your data is shiny, clean, and ready for modeling!")

        return X_train_final, y_train_balanced, X_test_final, y_test