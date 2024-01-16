import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel
import torch

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased')


class CategoricalDataProcessor:
    def __init__(self, data, categorical_features):
        """
        Initialize the CategoricalDataProcessor with the dataset.
        :param data: DataFrame - The dataset containing categorical features.
        """
        self.data = data
        self.categorical_features = categorical_features
        self.lyrics = None


    def remove_unwanted_features(self):
        """
        Remove specific unwanted features from the data.
        """
        print("Columns before removal:", self.data.columns)
        features_to_remove = ['simple_title', 'spotify_link', 'spotify_id', 'video_link', 'analysis_url']
        for feature in features_to_remove:
            if feature in self.data.columns:
                self.data = self.data.drop(feature, axis=1)
                print("Columns after removal:", self.data.columns)

    def separate_lyrics(self):
        if 'lyrics' in self.data.columns:
            self.lyrics = self.data['lyrics']
            self.data = self.data.drop('lyrics', axis=1)

    def convert_to_string(self):
        """
        Convert all categorical columns to string type.
        """
        for feature in self.categorical_features:
            print(self.categorical_features)
            if feature in self.data.columns:
                print(feature)
                print(self.data.columns)
                self.data[feature] = self.data[feature].astype(str)

    def handle_missing_values(self):
        """
        Handle missing values in categorical data.
        """
        for feature in self.categorical_features:
            self.data.fillna('Unknown', inplace=True)

    def encode_categorical_data(self):
        """
        Encode categorical data.
        """
        print(self.data.dtypes)
        categorical_features = [column for column in self.data.columns if column in self.categorical_features]
        # categorical_features = self.data.select_dtypes(include=['object']).columns
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(self.data[categorical_features])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))

        # Drop original categorical columns and add encoded columns
        self.data = self.data.drop(categorical_features, axis=1)
        self.data = pd.concat([self.data, encoded_df], axis=1)

    # Function to process and get BERT embeddings for a piece of text
    def get_bert_embeddings(text):
        # Tokenize the text
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        # Get BERT embeddings
        outputs = model(**inputs)
        # Extract the embeddings (for example, the last hidden state)
        embeddings = outputs.last_hidden_state
        # Process embeddings (e.g., mean pooling, dimensionality reduction)
        # ...
        return processed_embeddings

    def process_data(self):
        """
        Process the data by applying all the processing steps.
        """
        # print(self.data.columns)
        self.remove_unwanted_features()
        self.separate_lyrics()
        self.convert_to_string()
        self.handle_missing_values()
        self.encode_categorical_data()
        self.process_lyrics()

    def reintegrate_lyrics(self, processed_lyrics):
        self.data['processed_lyrics'] = processed_lyrics
