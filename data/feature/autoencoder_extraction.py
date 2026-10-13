"""
AutoEncoder extraction data preprocessing utilities
"""

import pandas as pd
from sklearn.decomposition import PCA
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping


class AutoEncoderExtractionDataPreprocessor:
    def __init__(self, dataset_name: str):
        pass
        self.dataset_name = dataset_name

    @staticmethod
    def training_ae_extraction(train_data, n_components, preprocessing_method):
        # Build AE with Functional API
        input_layer = Input(shape=(train_data.shape[1],))
        x = Dense(128, activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)
        bottleneck = Dense(n_components, activation='relu', name='bottleneck')(x)
        x = Dense(64, activation='relu')(bottleneck)
        x = Dense(128, activation='relu')(x)
        output_layer = Dense(train_data.shape[1])(x)

        # Full AE model
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

        # Train AE
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        autoencoder.fit(train_data, train_data,
                        epochs=100,
                        batch_size=128,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=[early_stop],
                        verbose=0)

        # Encoder model
        encoder = Model(inputs=input_layer, outputs=bottleneck)
        encoder.save(f"weights/encoder_model_{preprocessing_method}_{n_components}_comp.keras")

        return

    @staticmethod
    def autoencoder_extraction(train_set, test_set, n_components):
        pca = PCA(n_components=n_components)
        train = pd.DataFrame(pca.fit_transform(train_set))
        test = pd.DataFrame(pca.transform(test_set))
        return train, test
