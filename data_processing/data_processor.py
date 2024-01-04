from numerical_data_processor import NumericalDataProcessor
from categorical_data_processor import CategoricalDataProcessor
import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path, encoding='ISO-8859-1')

    def process_data(self):
        """
        Process the data using NumericalDataProcessor and CategoricalDataProcessor.
        """
        # Splitting the data
        numerical_data = self.data.select_dtypes(include=['int64', 'float64'])
        categorical_data = self.data.select_dtypes(include=['object'])

        # Process numerical data
        numerical_processor = NumericalDataProcessor(numerical_data)
        numerical_processor.process_data()

        # Process categorical data
        categorical_processor = CategoricalDataProcessor(categorical_data)
        categorical_processor.process_data()

        # Combine processed data
        self.data = pd.concat([numerical_processor.data, categorical_processor.data], axis=1)
