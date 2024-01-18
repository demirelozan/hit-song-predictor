from data_processing.numerical_data_processor import NumericalDataProcessor
from data_processing.categorical_data_processor import CategoricalDataProcessor
import pandas as pd
import torch
import torch.optim as optim
import machine_learning.bert_lyrical_model
import time
import re


class DataProcessor:
    def __init__(self, data, numerical_features, categorical_features):
        self.data = data
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.dropped_rows_count = 0

    @staticmethod
    def clean_feature_names(dataframe):

        # Regular expression to find non-alphanumeric characters
        non_alphanumeric_regex = re.compile('[^a-zA-Z0-9]')

        clean_names = {col: non_alphanumeric_regex.sub('_', col) for col in dataframe.columns}
        dataframe = dataframe.rename(columns=clean_names)
        return dataframe

    def remove_unwanted_features(self):
        """
        Remove specific unwanted features from the data.
        """
        print("Columns before removal:", self.data.columns)
        features_to_remove = ['date', 'year', 'title', 'simple_title', 'artist', 'main_artist', 'spotify_link', 'spotify_id', 'video_link', 'analysis_url']
        for feature in features_to_remove:
            if feature in self.data.columns:
                self.data = self.data.drop(feature, axis=1)
                print("Columns after removal:", self.data.columns)

    def remove_missing_values(self):
        """
        Remove rows where numerical features are 'unknown' or NA, except for certain verbal values in 'Change'.
        """
        # Define verbal values in 'Change' to be retained
        retain_verbal_values = ['New', 'Re-Entry', 'Hot Shot Debut']

        # Handle 'Change' separately
        if 'Change' in self.numerical_features:
            change_column = self.data['Change'].apply(
                lambda x: x if x in retain_verbal_values else pd.to_numeric(x, errors='coerce'))
            self.data['Change'] = change_column

        # Convert other numerical features from 'unknown' to NA
        for feature in [f for f in self.numerical_features if f != 'Change']:
            self.data[feature] = pd.to_numeric(self.data[feature], errors='coerce')

        # Drop rows with NA values in numerical features
        initial_row_count = len(self.data)
        self.data = self.data.dropna(subset=self.numerical_features)
        final_row_count = len(self.data)
        self.dropped_rows_count = initial_row_count - final_row_count

    def handle_BERT_process(self, categorical_processor):

        train_loader, val_loader = categorical_processor.create_data_loaders()

        # Model Initialization and Configuration
        bert_model_name = 'bert-base-uncased'
        hidden_dim = 256
        output_dim = 1  # Adjust based on your task
        n_layers = 2
        bidirectional = True
        dropout = 0.5
        linear_hidden = 128

        model = machine_learning.bert_lyrical_model.BERTLyricalModel(bert_model_name, hidden_dim, output_dim, n_layers,
                                                                     bidirectional, dropout,
                                                                     linear_hidden)
        try:
            # Freeze BERT parameters to prevent them from being updated during training
            for name, param in model.named_parameters():
                if 'bert' in name:
                    param.requires_grad = False

            # Device Configuration
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            criterion = machine_learning.bert_lyrical_model.MARELoss()
            criterion = criterion.to(device)

            # Define Optimizer
            optimizer = optim.AdamW(model.parameters(), lr=5e-5)

            N_EPOCHS = 4
            SAVE_PATH = 'D:/Machine Learning Datasets/model.pt'
            best_valid_loss = float('inf')

            for epoch in range(N_EPOCHS):
                start_time = time.time()

                # Assuming train_iterator and valid_iterator are defined
                train_loss, train_acc = machine_learning.bert_lyrical_model.train(model, train_loader, optimizer,
                                                                                  criterion,
                                                                                  device)
                valid_loss, valid_acc = machine_learning.bert_lyrical_model.evaluate(model, val_loader, criterion,
                                                                                     device)

                end_time = time.time()
                epoch_mins, epoch_secs = machine_learning.bert_lyrical_model.epoch_time(start_time, end_time)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), SAVE_PATH)

                print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

                # After training, get embeddings from the BERT model
                embeddings = categorical_processor.get_bert_embeddings()

                # Reintegrate the BERT embeddings back into the data
                self.reintegrate_embeddings(embeddings)
        except Exception as e:
            print(f"An error occurred during BERT processing: {e}")
            # Handle the error or re-raise
            raise

    def reintegrate_embeddings(self, embeddings):
        # Convert embeddings tensor to DataFrame
        embeddings_df = pd.DataFrame(embeddings.cpu().numpy())

        # Ensure the index aligns with the original data
        embeddings_df.index = self.data.index

        # Concatenate the embeddings with the original data
        self.data = pd.concat([self.data, embeddings_df], axis=1)

    def check_nan_values(self, data):
        # Check for NaN values in the dataset
        nan_columns = data.columns[data.isna().any()].tolist()
        return nan_columns

    def process_data(self):
        """
        Process the data using NumericalDataProcessor and CategoricalDataProcessor.
        """
        try:
            self.remove_unwanted_features()
            self.remove_missing_values()
            print(f'Count of dropped rows: {self.dropped_rows_count}')
            self.data = self.clean_feature_names(self.data)

            # Process numerical data
            numerical_processor = NumericalDataProcessor(self.data, self.numerical_features)
            numerical_processor.process_data()
            print('Numerical Processing Complete')
            # print('Numerical Processor Data is: ' + numerical_processor.data.columns)
            nan_columns = self.check_nan_values(numerical_processor.data)
            print("Numerical Columns with NaN values: ", nan_columns)
            output_file_path = 'D:/Machine Learning Datasets/updated_data_after_numerical_processor.xlsx'
            if output_file_path:
                numerical_processor.data.to_excel(output_file_path, index=False)

            # Process categorical data
            categorical_processor = CategoricalDataProcessor(self.data, self.categorical_features)
            categorical_processor.process_data()
            print('Categorical Processing Complete')
            nan_columns = self.check_nan_values(categorical_processor.data)
            print("Categorical Columns with NaN values: ", nan_columns)
            # self.handle_BERT_process(categorical_processor)
            print('Categorical Processor Data is: ' + categorical_processor.data.columns)

            self.data = pd.concat([numerical_processor.data, categorical_processor.data], axis=1)
            nan_columns = self.check_nan_values(self.data)
            print("Columns with NaN values: ", nan_columns)

            # When continuing BERT, use the second one (that also adds self.data to pd.concat)
            # self.data = pd.concat([numerical_processor.data, categorical_processor.data, self.data], axis=1) # Combine the data

            return self.data
        except Exception as e:
            print(f"An error occurred during data processing: {e}")
            # Handle the error or re-raise
            raise
