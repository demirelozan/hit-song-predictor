import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class CategoricalDataProcessor:
    def __init__(self, data, categorical_features):
        """
        Initialize the CategoricalDataProcessor with the dataset.
        :param data: DataFrame - The dataset containing categorical features.
        """
        self.data = data[categorical_features]

    def handle_missing_values(self):
        """
        Handle missing values in categorical data.
        """
        # Example: Fill missing values with a placeholder
        self.data.fillna('Unknown', inplace=True)

    def encode_categorical_data(self):
        """
        Encode categorical data.
        """
        categorical_features = self.data.select_dtypes(include=['object']).columns
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(self.data[categorical_features])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names(categorical_features))

        # Drop original categorical columns and add encoded columns
        self.data = self.data.drop(categorical_features, axis=1)
        self.data = pd.concat([self.data, encoded_df], axis=1)

    def process_lyrics(self):
        """
        Process the lyrics for NLP tasks.
        """
        # Implement NLP techniques for lyrics processing
        # Example: Sentiment analysis, keyword extraction, etc.

    def process_data(self):
        """
        Process the data by applying all the processing steps.
        """
        self.handle_missing_values()
        self.encode_categorical_data()
        self.process_lyrics()