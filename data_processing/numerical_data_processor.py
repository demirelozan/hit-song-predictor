import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class NumericalDataProcessor:
    def __init__(self, data):
        """
        Initialize with the dataset containing numerical features.
        :param data: DataFrame - The dataset.
        """
        self.data = data

    def handle_missing_values(self):
        """
        Handle missing values in numerical data.
        """
        self.data.fillna(self.data.mean(), inplace=True)

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
        self.handle_missing_values()
        if normalize:
            self.normalize_features()
        else:
            self.standardize_features()

