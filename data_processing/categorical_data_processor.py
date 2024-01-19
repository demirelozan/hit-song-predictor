import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data_model.numerical_song_data import NumericalSongData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CategoricalDataProcessor:
    def __init__(self, data, categorical_features):
        """
        Initialize the CategoricalDataProcessor with the dataset.
        :param data: DataFrame - The dataset containing categorical features.
        """
        self.data = data
        self.categorical_features = categorical_features
        self.lyrics = None
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.numerical_features = NumericalSongData.get_numerical_features()

    def remove_numerical_features(self):
        for feature in self.numerical_features:
            if feature in self.data.columns:
                self.data = self.data.drop(feature, axis=1)
                print("Columns after removal:", self.data.columns)

    def separate_lyrics(self):
        if 'lyrics' in self.data.columns:
            self.lyrics = self.data[['lyrics']].copy()  # Keep lyrics in a separate DataFrame
            self.data = self.data.drop('lyrics', axis=1)  # Drop lyrics from the main DataFrame

    def convert_to_string(self):
        """
        Convert specific categorical columns to string type.
        """
        for feature in ['title']:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].astype(str)

    def handle_missing_values(self):
        """
        Handle missing values selectively for categorical data.
        """
        # For columns like 'broad_genre', replace NaN with a placeholder like 'Unknown'
        if 'broad_genre' in self.categorical_features:
            self.data['broad_genre'].fillna('Unknown', inplace=True)

    # Preprocess titles: Remove symbols, punctuation, short terms, stopwords

    def preprocess_song_titles(self):
        """
        Preprocess song titles for LDA.
        """

        def preprocess_title(title):
            title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
            title = ' '.join([word for word in title.split() if len(word) > 3])  # Remove short words
            # Add additional preprocessing steps if necessary
            return title

        self.data['title'] = self.data['title'].apply(preprocess_title)

    def apply_lda(self, n_components=10):
        """
        Apply LDA to the song titles.
        """
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.data['title'])
        lda = LatentDirichletAllocation(n_components=n_components, random_state=0)
        lda.fit(X)

        topic_results = lda.transform(X)
        self.data['song_title_topic'] = topic_results.argmax(axis=1)

    def numerical_encoding_for_genres(self):
        """
        Apply numerical encoding to genres.
        """
        genre_mapping = {'country': 1, 'edm': 2, 'pop': 3, 'r&b': 4, 'rock': 5, 'rap': 6}
        self.data['genre_class'] = self.data['broad_genre'].map(genre_mapping).fillna(
            0)  # Fill NaN with a default value

    def tokenize_lyrics(self, lyrics_data):
        input_ids = []
        attention_masks = []

        # Replace NaN and specific error messages with a placeholder
        lyrics_data = lyrics_data.fillna("No Lyrics").replace("Error: Could not find lyrics.", "No Lyrics")

        for lyric in lyrics_data:
            # Skip empty strings or placeholders, if desired
            if isinstance(lyric, float) or lyric.strip() == "" or lyric.strip() == "No Lyrics":
                continue

            encoded_dict = self.tokenizer.encode_plus(
                lyric,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

    def get_bert_embeddings(self, data_subset):
        lyrics_data = data_subset['lyrics']  # Extract lyrics from the provided dataset
        input_ids, attention_masks = self.tokenize_lyrics(lyrics_data)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_masks)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings

    def create_data_loaders(self, batch_size=32, test_size=0.2):
        # Split data into training and validation sets first
        train_lyrics, val_lyrics, train_data, val_data = train_test_split(
            self.lyrics, self.data, test_size=test_size, stratify=self.data['Is Hit'])

        # Process and get embeddings for each subset
        train_embeddings = self.get_bert_embeddings(train_lyrics)
        val_embeddings = self.get_bert_embeddings(val_lyrics)

        train_targets = torch.tensor(train_data['Is Hit'].values).to(device)
        val_targets = torch.tensor(val_data['Is Hit'].values).to(device)

        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(train_embeddings, train_targets)
        val_dataset = TensorDataset(val_embeddings, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def process_data(self):
        """
        Process the data by applying all the processing steps.
        """
        # print(self.data.columns)
        self.remove_numerical_features()

        self.separate_lyrics()
        self.convert_to_string()
        self.handle_missing_values()

        self.preprocess_song_titles()
        self.apply_lda(n_components=10)
        # self.create_data_loaders()

        self.numerical_encoding_for_genres()

    def reintegrate_lyrics(self, processed_lyrics):
        self.data['processed_lyrics'] = processed_lyrics
