import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_model.categorical_song_data import CategoricalSongData


class NumericalDataProcessor:
    def __init__(self, data, numerical_features):
        """
        Initialize with the dataset containing numerical features.
        :param data: DataFrame - The dataset.
        """
        self.data = data
        self.numerical_features = numerical_features
        self.categorical_features = CategoricalSongData.get_categorical_features()

    def remove_categorical_features(self):
        for feature in self.categorical_features:
            if feature in self.data.columns:
                self.data = self.data.drop(feature, axis=1)
                # print("Columns after removal:", self.data.columns)

    def inspect_non_numeric_values(self):
        for feature in self.numerical_features:
            non_numeric = self.data[feature].apply(lambda x: not str(x).replace('.', '', 1).isdigit())
            if non_numeric.any():
                print(f'Non-numeric values found in {feature}:', self.data[feature][non_numeric].unique())

    def normalize_features(self):
        """
        Normalize numerical features.
        """
        scaler = MinMaxScaler()
        self.data[self.numerical_features] = scaler.fit_transform(self.data[self.numerical_features])

    def standardize_features(self):
        """
        Standardize numerical features.
        """
        scaler = StandardScaler()
        self.data[self.numerical_features] = scaler.fit_transform(self.data[self.numerical_features])

    def process_data(self, normalize=True):
        """
        Process the data by applying all the processing steps.
        :param normalize: bool - Whether to normalize (True) or standardize (False) the data.
        """

        self.remove_categorical_features()
        # Debugging before convert_types_and_handle_missing
        print("Before handling missing values:")
        print(self.data.isnull().sum())

        self.inspect_non_numeric_values()

        # Debugging after convert_types_and_handle_missing
        print("After handling missing values:")
        print(self.data.isnull().sum())
        if normalize:
            self.normalize_features()
        else:
            self.standardize_features()
