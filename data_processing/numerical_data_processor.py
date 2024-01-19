import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_model.numerical_song_data import NumericalSongData
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
        self.spotify_numerical_features = NumericalSongData.get_spotify_numerical_features()
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
        Normalize numerical features except for spotify_numerical_features.
        :param spotify_numerical_features: list - List of Spotify numerical features to exclude from normalization.
        """
        # Select features to normalize
        features_to_normalize = [f for f in self.numerical_features if f not in self.spotify_numerical_features]

        # Apply MinMaxScaler only to the selected features
        scaler = MinMaxScaler()
        self.data[features_to_normalize] = scaler.fit_transform(self.data[features_to_normalize])

    def standardize_features(self):
        """
        Standardize numerical features.
        """
        features_to_standardize = [f for f in self.numerical_features if f not in self.spotify_numerical_features]

        scaler = StandardScaler()
        self.data[features_to_standardize] = scaler.fit_transform(self.data[features_to_standardize])

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
