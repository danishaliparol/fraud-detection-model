Fraud Detection Model: End-to-End Pipeline with Real-time API
This repository provides a comprehensive solution for building, deploying, and serving a machine learning-powered fraud detection system. It covers data preprocessing, handling class imbalance, model training with hyperparameter tuning, model persistence, and a real-time prediction API using Flask and Gunicorn.

Table of Contents
Features

Supported Python Version

Installation

Project Structure

Usage

Data Preparation

Step 1: Train and Save the Model

Step 2: Run the Real-time Prediction API

Step 3: Test the API Endpoint

Dockerization

Model Details

Preprocessing

Imbalance Handling Strategies

Machine Learning Models & Hyperparameter Tuning

Model Persistence

Evaluation Metrics

Troubleshooting Common Issues

Contributing

License

Features
Data Loading from CSV: Reads training and scoring data directly from local CSV files.

ID Column Handling: Automatically identifies and removes specified identifier columns (transaction_id, customer_id, etc.) from datasets before training and prediction. These IDs are preserved and included in the API's output.

Robust Data Preprocessing:

Missing Value Imputation: Uses K-Nearest Neighbors (KNN) imputation for numerical features and most-frequent imputation for categorical features.

Feature Scaling: Applies StandardScaler to numerical features.

Dynamic Categorical Encoding: Intelligently applies OneHotEncoder for low-cardinality categorical features (< 10 unique values) and OrdinalEncoder for high-cardinality features (>= 10 unique values). Both encoders handle unseen categories gracefully in new data.

Comprehensive Class Imbalance Strategies: Evaluates and compares different approaches:

No sampling

SMOTE (Synthetic Minority Oversampling Technique) only

Random Undersampling only

A combination of SMOTE and Random Undersampling

Multiple Machine Learning Models: Benchmarks the performance of:

Logistic Regression

Random Forest Classifier

XGBoost Classifier (eXtreme Gradient Boosting)

Hyperparameter Tuning: Utilizes GridSearchCV with StratifiedKFold cross-validation to find optimal hyperparameters for each model and sampling strategy combination, optimizing for F1-score.

XGBoost Class Weighting: Incorporates tuning of scale_pos_weight for XGBClassifier to explicitly address class imbalance during its training.

Best Model Selection: Automatically identifies the overall best model and sampling strategy based on the highest F1-score achieved.

Model Persistence: Saves the entire trained model pipeline (including preprocessor and best estimator) as a .pkl file using joblib.

Real-time Prediction API: A Flask application served by Gunicorn provides a RESTful endpoint for low-latency predictions on new transaction data.

Docker Support: Includes a Dockerfile for easy containerization and consistent deployment across environments.

Supported Python Version
This project is specifically designed and tested with Python 3.12.4. It is highly recommended to use this exact version for consistent local development and production deployments to avoid dependency conflicts.

Installation
Clone the repository:

git clone [https://github.com/danishaliparol/fraud-detection-model.git](https://github.com/danishaliparol/fraud-detection-model.git)
cd fraud-detection-model

Ensure Python 3.12.4 is Installed:
Verify your Python version:

python --version
# or
python3.12 --version

If not installed, download and install Python 3.12.4 from python.org. Ensure it's added to your system's PATH.

Create a Python 3.12.4 virtual environment (recommended for managing dependencies):

# On Windows (Command Prompt/PowerShell - use 'py -3.12' if 'python' isn't 3.12):
py -3.12 -m venv .venv
# On macOS/Linux (Bash/Zsh - use 'python3.12' if 'python' isn't 3.12):
python3.12 -m venv .venv

This explicitly creates the virtual environment using your Python 3.12.4 executable.

Activate the virtual environment:

# On Windows (Command Prompt/PowerShell):
.\.venv\Scripts\activate
# On macOS/Linux (Bash/Zsh):
source ./.venv/bin/activate

You should see (.venv) in your terminal prompt.

Install dependencies from requirements.txt:

pip install -r requirements.txt

Project Structure
fraud_detection_project/
├── fraud_detection.py        # Main script for training, evaluation, batch prediction
├── api.py                    # Flask API for real-time predictions
├── sample_fraud_data.csv                 # Your training data (e.g., 10k rows)
├── data_to_be_predicted.csv                 # Your unseen data for scoring/prediction (e.g., a few rows for testing)
├── requirements.txt          # Python package dependencies
├── Dockerfile                # Instructions for building the Docker image
├── .gitignore                # Specifies files/folders to ignore from Git
├── best_fraud_model.pkl      # Generated after the first successful training run
└── README.md                 # This documentation file

Usage
Data Preparation
sample_fraud_data.csv:

Contains your training data.

Must include all feature columns, your target variable (e.g., is_fraud), and any ID columns (e.g., transaction_id, customer_id).

Important: Update target_column_name and id_columns variables in fraud_detection.py and api.py to precisely match your CSV headers.

Example sample_fraud_data.csv structure:

       'transaction_id', 'timestamp', 'amount', 'currency', 'merchant_id',
       'merchant_category', 'transaction_type', 'channel', 'location',
       'device_id', 'ip_address', 'customer_id', 'account_age_days',
       'customer_segment', 'credit_score', 'avg_transaction_amt',
       'txn_frequency_last_30d', 'txn_amt_last_24h', 'last_login_time',
       'home_location', 'device_type', 'os', 'browser', 'device_trust_score',
       'is_new_device', 'geo_distance_from_home', 'ip_risk_score',
       'unusual_time_flag', 'unusual_location_flag', 'velocity_flag',
       'geo_velocity_kmph', 'recent_failed_logins', 'device_change_count',
       'is_fraud'
...

data_to_be_predicted.csv:

Contains unseen data for which you want predictions.

Must have the same feature columns and ID columns as sample_fraud_data.csv, in the same order.

Must NOT contain the target column.

Example data_to_be_predicted.csv structure:

       'transaction_id', 'timestamp', 'amount', 'currency', 'merchant_id',
       'merchant_category', 'transaction_type', 'channel', 'location',
       'device_id', 'ip_address', 'customer_id', 'account_age_days',
       'customer_segment', 'credit_score', 'avg_transaction_amt',
       'txn_frequency_last_30d', 'txn_amt_last_24h', 'last_login_time',
       'home_location', 'device_type', 'os', 'browser', 'device_trust_score',
       'is_new_device', 'geo_distance_from_home', 'ip_risk_score',
       'unusual_time_flag', 'unusual_location_flag', 'velocity_flag',
       'geo_velocity_kmph', 'recent_failed_logins', 'device_change_count'
...

Place both sample_fraud_data.csv and data_to_be_predicted.csv in the root of your fraud_detection_project directory.

Step 1: Train and Save the Model
Run the main script to train the model, evaluate it, and save the best-performing pipeline.

python fraud_detection.py

First run: The script will print messages indicating that no pre-trained model was found. It will load data from sample_fraud_data.csv, remove id_columns, proceed with training, evaluation, save best_fraud_model.pkl, and then make predictions on score.csv.

Subsequent runs: If best_fraud_model.pkl already exists, the script will load it and directly perform predictions on data_to_be_predicted.csv (if available), skipping retraining.

Step 2: Run the Real-time Prediction API
Once best_fraud_model.pkl is generated, you can start the API for real-time predictions.

Ensure api.py and requirements.txt are set up:

Make sure you have the api.py file in your project root.

Make sure requirements.txt contains Flask and gunicorn.

Crucially, ensure you have installed the dependencies from requirements.txt in your Python 3.12.4 virtual environment (steps for this are detailed in the Installation section).

Activate your virtual environment:
If you've closed your terminal or deactivated the environment, activate it again:

# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source ./.venv/bin/activate

Start the Gunicorn server:
In your terminal, within the activated virtual environment and your fraud_detection_project directory, run:

gunicorn -w 4 -b 0.0.0.0:8000 api:app

You should see messages from Gunicorn indicating workers starting and the Flask app loading the model. The API will be listening on http://127.0.0.1:8000.

Step 3: Test the API Endpoint
With the API running in one terminal, open another terminal or use a tool like Postman or Insomnia to send a POST request to the /predict endpoint.

API Endpoint: http://127.0.0.1:8000/predict
Method: POST
Content-Type: application/json

Example curl request:

curl -X POST \
  [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \
  -H 'Content-Type: application/json' \
  -d '[
    {
      "transaction_id": "TXN123456789",
      "timestamp": "2025-06-16T10:45:00Z",
      "amount": 253.75,
      "currency": "USD",
      "merchant_id": "M987654",
      "merchant_category": "Electronics",
      "transaction_type": "Online",
      "channel": "Mobile",
      "location": "Riyadh, SA",
      "device_id": "DEV12345XYZ",
      "ip_address": "192.168.1.42",
      "customer_id": "CUST001122",
      "account_age_days": 420,
      "customer_segment": "Gold",
      "credit_score": 720,
      "avg_transaction_amt": 187.6,
      "txn_frequency_last_30d": 15,
      "txn_amt_last_24h": 460.9,
      "last_login_time": "2025-06-16T09:55:00Z",
      "home_location": "Jeddah, SA",
      "device_type": "Smartphone",
      "os": "Android",
      "browser": "Chrome",
      "device_trust_score": 0.87,
      "is_new_device": 0,
      "geo_distance_from_home": 845.2,
      "ip_risk_score": 0.32,
      "unusual_time_flag": 0,
      "unusual_location_flag": 1,
      "velocity_flag": 0,
      "geo_velocity_kmph": 98.4,
      "recent_failed_logins": 1,
      "device_change_count": 0
    },
    {
      "transaction_id": "TXN123456768",
      "timestamp": "2025-06-16T10:45:00Z",
      "amount": 253.75,
      "currency": "USD",
      "merchant_id": "M987654",
      "merchant_category": "Electronics",
      "transaction_type": "Online",
      "channel": "Mobile",
      "location": "Riyadh, SA",
      "device_id": "DEV12345XYZ",
      "ip_address": "192.168.1.42",
      "customer_id": "CUST001125",
      "account_age_days": 120,
      "customer_segment": "Gold",
      "credit_score": 460,
      "avg_transaction_amt": 187.6,
      "txn_frequency_last_30d": 15,
      "txn_amt_last_24h": 460.9,
      "last_login_time": "2025-06-16T09:55:00Z",
      "home_location": "Jeddah, SA",
      "device_type": "Smartphone",
      "os": "Android",
      "browser": "Chrome",
      "device_trust_score": 0.21,
      "is_new_device": 0,
      "geo_distance_from_home": 10000.2,
      "ip_risk_score": 0.88,
      "unusual_time_flag": 0,
      "unusual_location_flag": 1,
      "velocity_flag": 0,
      "geo_velocity_kmph": 22.1,
      "recent_failed_logins": 1,
      "device_change_count": 0
    }
  ]'

Expected JSON Response:

[
  {
    "transaction_id": 10001,
    "customer_id": 501,
    "predicted_is_fraud": 0
  },
  {
    "transaction_id": 10002,
    "customer_id": 502,
    "predicted_is_fraud": 1
  }
]

Dockerization
To containerize this application for consistent deployment:

Build the Docker Image:
In your project root directory (where Dockerfile is located), run:

docker build -t fraud-detection-app .

This command builds a Docker image named fraud-detection-app.

Run the Docker Container:
To run the Flask API inside a Docker container, mapping port 8000 from the container to your host machine:

docker run -p 8000:8000 fraud-detection-app

You can then test the API using curl as shown in Step 3: Test the API Endpoint.

For production deployment, consider pushing your image to a container registry (like Docker Hub, Google Container Registry, AWS ECR) and deploying to a cloud platform (e.g., Google Cloud Run, AWS Fargate, Kubernetes).

Model Details
Preprocessing
The _create_preprocessor method dynamically sets up a ColumnTransformer to handle different feature types:

Numerical Imputation & Scaling: KNNImputer(n_neighbors=5) for missing values, followed by StandardScaler for all numerical features.

Categorical Imputation & Encoding:

SimpleImputer(strategy='most_frequent') fills missing categorical values.

Low Cardinality (< 10 unique values): OneHotEncoder(handle_unknown='ignore') is used, which outputs zeros for any new, unseen categories during prediction.

High Cardinality (>= 10 unique values): OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) is used, assigning a specific -1 value to new, unseen categories.

Imbalance Handling Strategies
The train function evaluates how effectively different sampling strategies (applied within imblearn.pipeline.Pipeline) improve model performance on imbalanced data:

None: Baseline with no resampling applied.

SMOTE_Only: SMOTE(sampling_strategy=0.1) to synthesize new minority class samples.

Undersampling_Only: RandomUnderSampler(sampling_strategy=0.5) to reduce the number of majority class samples.

SMOTE_Undersampling: A sequential combination of SMOTE followed by Random Undersampling.

Machine Learning Models & Hyperparameter Tuning
The script evaluates the following classification models for each sampling strategy:

Logistic Regression: Hyperparameters C (regularization strength) and penalty (l1 or l2) are tuned. Supports class_weight.

Random Forest Classifier: Tuned for n_estimators (number of trees) and max_depth. Supports class_weight.

XGBoost Classifier: Tuned for n_estimators, learning_rate, max_depth, subsample, colsample_bytree. Crucially, its scale_pos_weight parameter is also tuned using a range around the calculated ratio of negative to positive samples, providing explicit class weighting.

GridSearchCV with StratifiedKFold (3 splits) is employed for exhaustive hyperparameter search and robust cross-validation. The F1-score is the primary metric for optimization and best model selection due to its effectiveness in balancing precision and recall for imbalanced datasets.

Model Persistence
The FraudDetectionModel instance, which internally holds the best preprocessing steps, chosen sampling strategy, and the trained classifier, is serialized to best_fraud_model.pkl using Python's joblib library. This serialized file allows:

Fast Reloading: Rapid deployment of the trained model without needing to retrain.