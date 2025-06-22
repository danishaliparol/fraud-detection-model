import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import os
import joblib
import xgboost as xgb

# Suppress specific warnings for cleaner output in example
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='imblearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

class FraudDetectionModel:
    """
    An end-to-end fraud detection model pipeline including preprocessing,
    handling imbalanced data, hyperparameter tuning, and prediction.
    """

    def __init__(self):
        self.preprocessor = None
        self.best_model = None
        self.best_f1_score = -1
        self.model_name = None
        self.best_sampling_strategy = None

    def _create_preprocessor(self, X):
        """
        Creates a ColumnTransformer for preprocessing numerical and categorical features.
        - Numerical features: KNN Imputation, then StandardScaler.
        - Categorical features: Most Frequent Imputation, then OneHotEncoder (low cardinality)
                                or OrdinalEncoder (high cardinality), handling unseen.
        """
        numerical_features = X.select_dtypes(include=np.number).columns
        categorical_features = X.select_dtypes(include='object').columns

        numerical_transformer = Pipeline(steps=[
            ('knn_imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])

        categorical_transformers = []
        for col in categorical_features:
            if X[col].nunique() < 10:
                categorical_transformers.append(
                    (f'onehot_{col}',
                     Pipeline(steps=[
                         ('freq_imputer', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore'))
                     ]),
                     [col])
                )
            else:
                categorical_transformers.append(
                    (f'label_{col}',
                     Pipeline(steps=[
                         ('freq_imputer', SimpleImputer(strategy='most_frequent')),
                         ('label', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                     ]),
                     [col])
                )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
            ] + categorical_transformers,
            remainder='passthrough'
        )

    def train(self, X_train, y_train):
        """
        Trains the fraud detection model.
        Includes preprocessing, evaluation of different sampling strategies (None, SMOTE, Undersampling, SMOTE+Undersampling),
        hyperparameter tuning, and model selection based on F1 score.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
        """
        print("Starting model training with multiple sampling strategies...")

        self._create_preprocessor(X_train)

        # Calculate initial scale_pos_weight for reference and tuning
        neg_count = y_train.value_counts()[0] if 0 in y_train.value_counts() else 1
        pos_count = y_train.value_counts()[1] if 1 in y_train.value_counts() else 1
        initial_scale_pos_weight = neg_count / pos_count
        print(f"Calculated initial scale_pos_weight: {initial_scale_pos_weight:.2f}")


        models = {
            'LogisticRegression': {
                'estimator': LogisticRegression(random_state=42, solver='liblinear'),
                'param_grid': {
                    'estimator__C': [0.1, 1, 10],
                    'estimator__penalty': ['l1', 'l2'],
                    'estimator__class_weight': [None, 'balanced']
                }
            },
            'RandomForestClassifier': {
                'estimator': RandomForestClassifier(random_state=42),
                'param_grid': {
                    'estimator__n_estimators': [50, 100, 200],
                    'estimator__max_depth': [5, 10, None],
                    'estimator__class_weight': [None, 'balanced']
                }
            },
            'XGBClassifier': {
                'estimator': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42),
                'param_grid': {
                    'estimator__n_estimators': [50, 100, 200],
                    'estimator__learning_rate': [0.01, 0.1, 0.2],
                    'estimator__max_depth': [3, 5, 7],
                    'estimator__subsample': [0.8, 1.0],
                    'estimator__colsample_bytree': [0.8, 1.0],
                    'estimator__scale_pos_weight': [1, initial_scale_pos_weight, initial_scale_pos_weight * 0.5, initial_scale_pos_weight * 2]
                }
            }
        }

        sampling_strategies = {
            'None': [],
            'SMOTE_Only': [('oversampler', SMOTE(sampling_strategy=0.1, random_state=42))],
            'Undersampling_Only': [('undersampler', RandomUnderSampler(sampling_strategy=0.5, random_state=42))],
            'SMOTE_Undersampling': [
                ('oversampler', SMOTE(sampling_strategy=0.1, random_state=42)),
                ('undersampler', RandomUnderSampler(sampling_strategy=0.5, random_state=42))
            ]
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for sampling_name, sampling_steps in sampling_strategies.items():
            print(f"\n--- Evaluating with Sampling Strategy: {sampling_name} ---")
            for model_name, config in models.items():
                print(f"\nTraining and tuning {model_name}...")

                pipeline_steps = [('preprocessor', self.preprocessor)] + sampling_steps + [('estimator', config['estimator'])]
                pipeline = ImbPipeline(steps=pipeline_steps)

                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=config['param_grid'],
                    scoring='f1',
                    cv=cv,
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(X_train, y_train)

                current_f1 = grid_search.best_score_
                print(f"Best F1 Score for {model_name} with {sampling_name}: {current_f1:.4f}")

                if current_f1 > self.best_f1_score:
                    self.best_f1_score = current_f1
                    self.best_model = grid_search.best_estimator_
                    self.model_name = model_name
                    self.best_sampling_strategy = sampling_name

        print(f"\nTraining complete. Overall best combination:")
        print(f"  Model: {self.model_name}")
        print(f"  Sampling Strategy: {self.best_sampling_strategy}")
        print(f"  Achieved F1 Score: {self.best_f1_score:.4f}")

    def predict(self, X_unseen):
        """
        Predicts on unseen data using the best trained model.

        Args:
            X_unseen (pd.DataFrame): Unseen features to predict on.

        Returns:
            np.array: Predicted labels (0 for non-fraud, 1 for fraud).
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        print(f"Predicting on unseen data using the best model ({self.model_name} with {self.best_sampling_strategy})...")
        predictions = self.best_model.predict(X_unseen)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluates the best trained model on a test set.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for the test set.

        Returns:
            None: Prints classification report and confusion matrix.
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        print(f"\nEvaluating the best model ({self.model_name} with {self.best_sampling_strategy}) on the test set...")
        y_pred = self.best_model.predict(X_test)

        # Print detailed classification metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Print the confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))


        # Example Usage:
if __name__ == "__main__":
    # --- IMPORTANT: Update these file paths to your local CSV files ---
    train_file_path = 'sample_fraud_data.csv'
    score_file_path = 'data_to_be_predicted.csv'
    target_column_name = 'is_fraud' # Make sure this matches the name of your target column in train.csv
    # Columns to be ignored from training/prediction (e.g., identifiers)
    id_columns = ['transaction_id', 'timestamp', 'currency', 'merchant_id', 'device_id',
                  'customer_id', 'last_login_time']
    model_file_name = 'best_fraud_model.pkl'

    fraud_detector = FraudDetectionModel()

    # --- Check if a trained model already exists ---
    if os.path.exists(model_file_name):
        print(f"Pre-trained model '{model_file_name}' found. Loading model for prediction...")
        try:
            fraud_detector = joblib.load(model_file_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading the model file: {e}. Please ensure it's a valid pickle file.")
            print("Please delete the corrupted file or provide a valid one and re-run.")
            exit()

        try:
            X_unseen_raw = pd.read_csv(score_file_path)
            print(f"Scoring data loaded from '{score_file_path}'.")

            # Remove ID columns from unseen data before prediction
            X_unseen = X_unseen_raw.copy() # Work on a copy to avoid modifying original dataframe if needed elsewhere
            for col in id_columns:
                if col in X_unseen.columns:
                    X_unseen = X_unseen.drop(columns=[col])
                    print(f"Removed '{col}' from unseen data.")
                else:
                    print(f"Warning: ID column '{col}' not found in unseen data. Skipping removal.")

        except FileNotFoundError:
            print(f"Error: Scoring data file '{score_file_path}' not found. Cannot perform prediction.")
            exit()

        print("\n--- Making predictions on unseen data ---")
        predictions_unseen = fraud_detector.predict(X_unseen)
        print("\nPredictions on unseen data:")
        print(predictions_unseen)
        print("\nUnseen Data with Predictions:")
        X_unseen_raw['predicted_is_fraud'] = predictions_unseen # Add predictions back to the original raw data
        X_unseen_raw.to_csv('predicted_data.csv', index=False)
        print(X_unseen_raw.head()) # Print head to show new column with predictions

    else:
        print(f"No pre-trained model '{model_file_name}' found. Starting model training...")
        try:
            data_train_raw = pd.read_csv(train_file_path)
            print(f"Training data loaded from '{train_file_path}'.")

            # Remove ID columns from training data before splitting features/target
            data_train = data_train_raw.copy()
            for col in id_columns:
                if col in data_train.columns:
                    data_train = data_train.drop(columns=[col])
                    print(f"Removed '{col}' from training data.")
                else:
                    print(f"Warning: ID column '{col}' not found in training data. Skipping removal.")


        except FileNotFoundError:
            print(f"Error: Training data file '{train_file_path}' not found. Cannot proceed with training.")
            exit()

        if target_column_name not in data_train.columns:
            print(f"Error: Target column '{target_column_name}' not found in '{train_file_path}'. Please check the column name.")
            exit()
        X = data_train.drop(target_column_name, axis=1)
        y = data_train[target_column_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        print("Dataset loaded and split with the following class distribution in training data:")
        print(y_train.value_counts())
        print("-" * 50)

        fraud_detector.train(X_train, y_train)

        fraud_detector.evaluate(X_test, y_test)

        print(f"\nSaving the best trained model to '{model_file_name}'...")
        try:
            joblib.dump(fraud_detector, model_file_name)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving the model: {e}")

        print("\n--- Attempting to load scoring data for prediction ---")
        try:
            X_unseen_raw = pd.read_csv(score_file_path)
            print(f"Scoring data loaded from '{score_file_path}'.")

            X_unseen = X_unseen_raw.copy()
            for col in id_columns:
                if col in X_unseen.columns:
                    X_unseen = X_unseen.drop(columns=[col])
                    print(f"Removed '{col}' from unseen data before prediction.")
                else:
                    print(f"Warning: ID column '{col}' not found in unseen data. Skipping removal.")

            predictions_unseen = fraud_detector.predict(X_unseen)
            print("\nPredictions on unseen data:")
            print(predictions_unseen)
            print("\nUnseen Data with Predictions:")
            X_unseen_raw['predicted_is_fraud'] = predictions_unseen
            X_unseen_raw.to_csv('predicted_data.csv', index=False)
            print(X_unseen_raw.head())
        except FileNotFoundError:
            print(f"Warning: Scoring data file '{score_file_path}' not found. Cannot perform prediction on unseen data now.")
            print("The model has been trained and saved, but no predictions were made due to missing scoring data.")