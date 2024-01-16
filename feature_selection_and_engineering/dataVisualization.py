import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataVisualization:
    def __init__(self, data):
        self.data = data

    def plot_scatter(self, variable, title_prefix, figsize=(10, 6)):
        plt.figure(figsize=figsize)
        sns.scatterplot(data=self.data, x=variable, y='Is Hit', alpha=0.5)
        plt.title(f'{title_prefix} {variable} vs Is Hit')
        plt.ylabel('Is Hit')
        plt.xlabel(variable)
        plt.show()

    # Generate a Heatmap to show correlations with revenue
    def plot_heatmap(self, figsize=(12, 8)):
        plt.figure(figsize=figsize)
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix[['Is Hit']].sort_values(by='Is Hit', ascending=False),
                    annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation of Features with Is Hit')
        plt.show()

    def plot_correlation_matrix(self):
        # Encode categorical variables
        encoded_data = self.data.copy()
        for column in encoded_data.select_dtypes(include=['object']).columns:
            encoded_data[column] = LabelEncoder().fit_transform(encoded_data[column])

        # Plot correlation matrix
        plt.figure(figsize=(15, 10))
        sns.heatmap(encoded_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix for All Variables')
        plt.show()

    def plot_pairplot(self, selected_columns, figsize=(12, 16), aspect=0.74, plot_kws={'s': 10}):
        """Plot pairwise relationships for selected continuous variables with adjusted plot sizes."""
        sns.set(style="ticks", font_scale=0.75)
        g = sns.pairplot(self.data[selected_columns], height=figsize[0] / len(selected_columns), aspect=aspect,
                         plot_kws=plot_kws)
        g.fig.set_size_inches(*figsize)

        # Rotate x-axis labels and adjust layout
        for ax in g.axes.flatten():
            plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

        plt.tight_layout()
        plt.show()
        sns.reset_orig()

    def plot_selected_features_correlation(self, feature_list):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data[feature_list].corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix for Selected Features')
        plt.show()

    def execute_visualizations(self, numeric_feature_columns):

        print("First few rows of data:")
        print(self.data.head())

        print("\nMissing values in each column:")
        print(self.data.isnull().sum())

        print("\nData types of columns:")
        print(self.data.dtypes)
        # Call the visualization functions
        for column in numeric_feature_columns:
            self.plot_scatter(column, 'Scatter Plot of', figsize=(10, 6))

        # self.plot_heatmap(figsize=(12, 8))

        self.plot_pairplot(numeric_feature_columns, figsize=(12, 16))

        # selected_features = ['SelectedFeature1', 'SelectedFeature2']  # Replace with your selected features
        # self.plot_selected_features_correlation(selected_features)

'''
    def main(self, test_data):
        self.execute_visualizations(numeric_feature_columns=['no'], test_data=test_data)


test_data_ = pd.DataFrame({'feature': [1, 2, 3, 4, 5], 'Is Hit': [0, 1, 0, 1, 0]})
dataVisualization = DataVisualization(test_data_)
dataVisualization.main(test_data_)
'''