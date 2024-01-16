import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class NumericalDataProcessor:
    def __init__(self, data, numerical_features):
        """
        Initialize with the dataset containing numerical features.
        :param data: DataFrame - The dataset.
        """
        self.data = data
        self.numerical_features = numerical_features

    def convert_types_and_handle_missing(self):
        """
        Convert columns to their appropriate numerical types and handle missing values.
        """
        for feature in self.numerical_features:
            print(feature)
            # Convert to numeric, coercing errors (like 'NA', 'unknown') to NaN
            self.data.loc[:, feature] = pd.to_numeric(self.data[feature], errors='coerce')

        # Calculate mean for numeric columns only and fill NaNs in those columns
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())

    def normalize_features(self):
        """
        Normalize numerical features.
        """
        scaler = MinMaxScaler()
        numerical_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])

    def standardize_features(self):
        """
        Standardize numerical features.
        """
        scaler = StandardScaler()
        numerical_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])

    def process_data(self, normalize=True):
        """
        Process the data by applying all the processing steps.
        :param normalize: bool - Whether to normalize (True) or standardize (False) the data.
        """
        self.convert_types_and_handle_missing()
        if normalize:
            self.normalize_features()
        else:
            self.standardize_features()
