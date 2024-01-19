import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelTraining:
    def __init__(self, data, target_column='is_hit'):
        self.feature_names = None
        self.X_test_reduced = None
        self.X_train_reduced = None
        self.preprocessor = None
        self.data = data
        self.target_column = target_column
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def separate_features_and_target(self):
        # Exclude specific features from the training data
        excluded_features = ['title', 'artist', 'peak_pos', 'genre', 'broad_genre']
        billboard_metadata_features = ['last_pos', 'weeks', 'rank', 'change']
        X = self.data.drop(self.target_column, axis=1)
        X = X.drop(excluded_features, axis=1, errors='ignore')  # Drop the columns, ignore if not present
        X = X.drop(billboard_metadata_features, axis=1, errors='ignore')
        self.feature_names = X.columns
        y = self.data[self.target_column]
        return X, y

    def split_data(self, test_size=0.20, random_state=42):
        X, y = self.separate_features_and_target()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    def check_nan_values(self):
        # Check for NaN values in the dataset
        nan_columns = self.data.columns[self.data.isna().any()].tolist()
        return nan_columns

    def perform_cross_validation(self, model, cv_folds=5):
        mse_scores = cross_val_score(model, self.X_train, self.y_train, scoring='neg_mean_squared_error', cv=cv_folds)
        r2_scores = cross_val_score(model, self.X_train, self.y_train, scoring='r2', cv=cv_folds)

        rmse_scores = np.sqrt(-mse_scores)
        avg_rmse = np.mean(rmse_scores)
        avg_r2 = np.mean(r2_scores)

        return {'Average RMSE': avg_rmse, 'Average R^2': avg_r2}

    def feature_selection_based_on_importance(self, model, threshold=0.01):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            feature_names = self.preprocessor.get_feature_names_out()
            selected_feature_indices = [i for i, imp in enumerate(model.feature_importances_) if imp >= threshold]

            # Filter the data
            self.X_train_reduced = self.X_train[:, selected_feature_indices]
            self.X_test_reduced = self.X_test[:, selected_feature_indices]

            selected_features = [feature_names[i] for i in selected_feature_indices]
            print("Selected features based on importance:", selected_features)
            return self.X_train_reduced, self.X_test_reduced, selected_features
        else:
            print("Feature importance not available for this model type.")
            return self.X_train, self.X_test, []

    def train_linear_regression(self):
        logging.info("Training Linear Regression model.")
        model = LinearRegression()
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_decision_tree(self):
        logging.info("Training Decision Tree model.")
        model = DecisionTreeRegressor(random_state=42)
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_knn(self):
        logging.info("Training k-Nearest Neighbors model with k=1.")
        model = KNeighborsClassifier(n_neighbors=1)
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_naive_bayes(self):
        logging.info("Training Naïve Bayes model with var_smoothing=0.031.")
        model = GaussianNB(var_smoothing=0.031)
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_random_forest(self):
        logging.info("Training Random Forest Classifier with gini index and 600 trees.")
        model = RandomForestClassifier(criterion='gini', n_estimators=600, random_state=42)
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_xgboost(self):
        logging.info("Training XGBoost model.")
        model = XGBClassifier(random_state=42)
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_logistic_regression(self):
        logging.info("Training Logistic Regression with liblinear solver and l1 penalty.")
        model = LogisticRegression(solver='liblinear', penalty='l1', C=1 / 3, random_state=42)
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_neural_network(self):
        logging.info("Training Neural Network (MLP) with specified parameters.")
        hidden_layer_sizes = (22, 22, 22, 22)

        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                              max_iter=4500,
                              random_state=42)

        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_gradient_boosting(self):
        logging.info("Training Gradient Boost Model.")
        model = GradientBoostingClassifier(random_state=42)
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def get_feature_importance(self, model):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Use the stored feature names
            feature_importances = pd.Series(importances, index=self.feature_names)
            return feature_importances.sort_values(ascending=False)
        else:
            return "Feature importance not available for this model type."

    def evaluate_model(self, model):
        predictions = model.predict(self.X_test)

        # Regression Metrics
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        r2 = r2_score(self.y_test, predictions)  # R-squared

        return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R^2": r2}

    def evaluate_model_classification(self, model):
        # Suppress warnings about feature names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            # Ensure the data is a NumPy array and C-contiguous
            X_test_array = np.ascontiguousarray(self.X_test.values)

            predictions = model.predict(X_test_array)

        # Classification Metrics
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions)
        auc_roc = roc_auc_score(self.y_test, predictions)

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC-ROC": auc_roc
        }

    def perform_models(self):
        nan_columns = self.check_nan_values()
        print("Columns with NaN values: ", nan_columns)
        self.split_data()

        lgr_model = self.train_logistic_regression()
        dt_model = self.train_decision_tree()
        knn_model = self.train_knn()
        nb_model = self.train_naive_bayes()
        mlp_model = self.train_neural_network()
        rf_model = self.train_random_forest()
        xgb_model = self.train_xgboost()
        gbm_model = self.train_gradient_boosting()

        lgr_feature_importance = self.get_feature_importance(lgr_model)
        print("Logistic Regression Feature Importance:\n", lgr_feature_importance)
        dt_feature_importance = self.get_feature_importance(dt_model)
        print("Decision Tree Feature Importance:\n", dt_feature_importance)
        knn_feature_importance = self.get_feature_importance(knn_model)
        print("k-Nearest Neighbors Feature Importance:\n", knn_feature_importance)
        nb_feature_importance = self.get_feature_importance(nb_model)
        print("Naive Bayes Feature Importance:\n", nb_feature_importance)
        mlp_feature_importance = self.get_feature_importance(mlp_model)
        print("Multilayer Perceptron Feature Importance: \n", mlp_feature_importance)
        rf_feature_importance = self.get_feature_importance(rf_model)
        print("Random Forest Feature Importance:\n", rf_feature_importance)
        xgb_feature_importance = self.get_feature_importance(xgb_model)
        print("Xgboost Feature Importance:\n", xgb_feature_importance)
        gbm_feature_importance = self.get_feature_importance(gbm_model)
        print("GBM Feature Importance:\n", gbm_feature_importance)

        # Evaluate Models
        print("Logistic Regression Accuracy: ", self.evaluate_model_classification(lgr_model))
        print("Decision Tree Accuracy:", self.evaluate_model_classification(dt_model))
        print("k-Nearest Neighbors Accuracy:", self.evaluate_model_classification(knn_model))
        print("Naive Bayes Accuracy: ", self.evaluate_model_classification(nb_model))
        print("Multilayer Perceptron Feature Importance: \n", self.evaluate_model_classification(mlp_model))
        print("Random Forest Accuracy:", self.evaluate_model_classification(rf_model))
        print("XGBoost Accuracy:", self.evaluate_model_classification(xgb_model))
        print("GBM Accuracy:", self.evaluate_model_classification(gbm_model))

        '''
        self.X_train, self.X_test, selected_features = self.feature_selection_based_on_importance(
            gbm_model)
        print("Selected Features:", selected_features)
        '''
