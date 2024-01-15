from data_processing.numerical_data_processor import NumericalDataProcessor
from data_processing.categorical_data_processor import CategoricalDataProcessor
import pandas as pd


class DataProcessor:
    def __init__(self, data, numerical_features, categorical_features):
        self.data = data
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

    def process_data(self):
        """
        Process the data using NumericalDataProcessor and CategoricalDataProcessor.
        """
        # Process numerical data
        numerical_processor = NumericalDataProcessor(self.data, self.numerical_features)
        numerical_processor.process_data()

        # Process categorical data
        categorical_processor = CategoricalDataProcessor(self.data, self.categorical_features)
        categorical_processor.process_data()

        # Combine processed data
        self.data = pd.concat([numerical_processor.data, categorical_processor.data], axis=1)
        return self.data