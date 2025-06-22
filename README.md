# Fraud Detection Model: End-to-End Pipeline with Real-time API

This repository provides a comprehensive solution for building, deploying, and serving a machine learning-powered fraud detection system. It covers data preprocessing, handling class imbalance, model training with hyperparameter tuning, model persistence, and a real-time prediction API using Flask and Gunicorn.

## Table of Contents

-   [Features](#features)
-   [Supported Python Version](#supported-python-version)
-   [Installation](#installation)
-   [Project Structure](#project-structure)
-   [Usage](#usage)
    -   [Data Preparation](#data-preparation)
    -   [Step 1: Train and Save the Model](#step-1-train-and-save-the-model)
    -   [Step 2: Run the Real-time Prediction API](#step-2-run-the-real-time-prediction-api)
    -   [Step 3: Test the API Endpoint](#step-3-test-the-api-endpoint)
-   [Dockerization](#dockerization)
-   [Model Details](#model-details)
    -   [Preprocessing](#preprocessing)
    -   [Imbalance Handling Strategies](#imbalance-handling-strategies)
    -   [Machine Learning Models & Hyperparameter Tuning](#machine-learning-models--hyperparameter-tuning)
    -   [Model Persistence](#model-persistence)
-   [Evaluation Metrics](#evaluation-metrics)
-   [Troubleshooting Common Issues](#troubleshooting-common-issues)
-   [Contributing](#contributing)
-   [License](#license)

## Features

-   **Data Loading from CSV**: Reads training and scoring data directly from local CSV files.
-   **ID Column Handling**: Automatically identifies and removes specified identifier columns (`transaction_id`, `customer_id`, etc.) from datasets before training and prediction. These IDs are preserved and included in the API's output.
-   **Robust Data Preprocessing**:
    -   **Missing Value Imputation**: Uses K-Nearest Neighbors (KNN) imputation for numerical features and most-frequent imputation for categorical features.
    -   **Feature Scaling**: Applies `StandardScaler` to numerical features.
    -   **Dynamic Categorical Encoding**: Intelligently applies `OneHotEncoder` for low-cardinality categorical features (< 10 unique values) and `OrdinalEncoder` for high-cardinality features (>= 10 unique values). Both encoders handle unseen categories gracefully in new data.
-   **Comprehensive Class Imbalance Strategies**: Evaluates and compares different approaches:
    -   No sampling
    -   SMOTE (Synthetic Minority Oversampling Technique) only
    -   Random Undersampling only
    -   A combination of SMOTE and Random Undersampling
-   **Multiple Machine Learning Models**: Benchmarks the performance of:
    -   Logistic Regression
    -   Random Forest Classifier
    -   XGBoost Classifier (eXtreme Gradient Boosting)
-   **Hyperparameter Tuning**: Utilizes `GridSearchCV` with `StratifiedKFold` cross-validation to find optimal hyperparameters for each model and sampling strategy combination, optimizing for **F1-score**.
-   **XGBoost Class Weighting**: Incorporates tuning of `scale_pos_weight` for `XGBClassifier` to explicitly address class imbalance during its training.
-   **Best Model Selection**: Automatically identifies the overall best model and sampling strategy based on the highest F1-score achieved.
-   **Model Persistence**: Saves the entire trained model pipeline (including preprocessor and best estimator) as a `.pkl` file using `joblib`.
-   **Real-time Prediction API**: A Flask application served by Gunicorn provides a RESTful endpoint for low-latency predictions on new transaction data.
-   **Docker Support**: Includes a `Dockerfile` for easy containerization and consistent deployment across environments.

## Supported Python Version

This project is specifically designed and tested with **Python 3.12.4**. It is highly recommended to use this exact version for consistent local development and production deployments to avoid dependency conflicts.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/fraud-detection-model.git](https://github.com/your-username/fraud-detection-model.git)
    cd fraud-detection-model
    ```
2.  **Ensure Python 3.12.4 is Installed**:
    Verify your Python version:
    ```bash
    python --version
    # or
    python3.12 --version
    ```
    If not installed, download and install Python 3.12.4 from [python.org](https://www.python.org/downloads/). Ensure it's added to your system's PATH.

3.  **Create a Python 3.12.4 virtual environment** (recommended for managing dependencies):
    ```bash
    # On Windows (Command Prompt/PowerShell - use 'py -3.12' if 'python' isn't 3.12):
    py -3.12 -m venv .venv
    # On macOS/Linux (Bash/Zsh - use 'python3.12' if 'python' isn't 3.12):
    python3.12 -m venv .venv
    ```
    *This explicitly creates the virtual environment using your Python 3.12.4 executable.*

4.  **Activate the virtual environment**:
    ```bash
    # On Windows (Command Prompt/PowerShell):
    .\.venv\Scripts\activate
    # On macOS/Linux (Bash/Zsh):
    source ./.venv/bin/activate
    ```
    You should see `(.venv)` in your terminal prompt.

5.  **Install dependencies** from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure
Great news that fraud_detection.py ran successfully! This means your model is trained, evaluated, and the best_fraud_model.pkl file has been created.

Now, let's get your project published on GitHub and then set up and run your real-time prediction API.

How to Publish This Project in GitHub
Publishing your project to GitHub involves initializing a Git repository, adding your files, committing them, and then pushing them to a new repository on GitHub.

Ensure Git is Installed:
If you don't have Git installed, download it from https://git-scm.com/.

Initialize Git Repository in Your Project Folder:

Open your terminal or command prompt.
Navigate to your fraud_detection_project directory (e.g., cd path/to/fraud_detection_project).
Initialize a new Git repository:
Bash

git init
Create a .gitignore file:
This file tells Git which files and folders to ignore (i.e., not track or upload to GitHub). It's crucial for keeping your repository clean and secure (e.g., avoiding uploading large model files or virtual environments).

In your fraud_detection_project folder, create a new file named .gitignore (make sure it starts with a dot).
Add the following content to .gitignore:
# Virtual environment directory
.venv/

# Python compiled files
__pycache__/
*.pyc

# Trained model file (can be large and is regenerated)
best_fraud_model.pkl

# Optionally, ignore your raw data files if they are large or sensitive.
# It's common to not commit raw data.
# train.csv
# score.csv
Save the .gitignore file.
Add Your Files to the Git Repository:
This stages all relevant files for the first commit.

Bash

git add .
Commit Your Code Locally:
This saves the current state of your files in your local Git history.

Bash

git commit -m "Initial commit: End-to-end fraud detection model with API and Docker setup"
Create a New Repository on GitHub:

Go to https://github.com/new in your web browser.
Give your new repository a name (e.g., fraud-detection-model).
Choose whether it's Public or Private.
Crucially: Do NOT check "Add a README file" or "Add .gitignore" here, as you've already created them locally.
Click "Create repository".
Link Your Local Repository to GitHub and Push Your Code:
After creating the repository on GitHub, you'll see a page with instructions. Copy the two lines under "…or push an existing local repository from the command line". They will look similar to this (replace your-username with your actual GitHub username and fraud-detection-model with your repository name):

Bash

git remote add origin https://github.com/your-username/fraud-detection-model.git
git branch -M main
git push -u origin main
Paste these commands one by one into your terminal and press Enter after each.
You might be prompted to enter your GitHub username and password/Personal Access Token.
Your project, including fraud_detection.py, api.py, requirements.txt, Dockerfile, and README.md (but excluding .venv/ and best_fraud_model.pkl due to .gitignore), is now published on GitHub!

Detailed README.md Format
Here is the comprehensive README.md content. You should create a file named README.md (no extension) in your fraud_detection_project folder and paste this content into it.

Markdown

# Fraud Detection Model: End-to-End Pipeline with Real-time API

This repository provides a comprehensive solution for building, deploying, and serving a machine learning-powered fraud detection system. It covers data preprocessing, handling class imbalance, model training with hyperparameter tuning, model persistence, and a real-time prediction API using Flask and Gunicorn.

## Table of Contents

-   [Features](#features)
-   [Supported Python Version](#supported-python-version)
-   [Installation](#installation)
-   [Project Structure](#project-structure)
-   [Usage](#usage)
    -   [Data Preparation](#data-preparation)
    -   [Step 1: Train and Save the Model](#step-1-train-and-save-the-model)
    -   [Step 2: Run the Real-time Prediction API](#step-2-run-the-real-time-prediction-api)
    -   [Step 3: Test the API Endpoint](#step-3-test-the-api-endpoint)
-   [Dockerization](#dockerization)
-   [Model Details](#model-details)
    -   [Preprocessing](#preprocessing)
    -   [Imbalance Handling Strategies](#imbalance-handling-strategies)
    -   [Machine Learning Models & Hyperparameter Tuning](#machine-learning-models--hyperparameter-tuning)
    -   [Model Persistence](#model-persistence)
-   [Evaluation Metrics](#evaluation-metrics)
-   [Troubleshooting Common Issues](#troubleshooting-common-issues)
-   [Contributing](#contributing)
-   [License](#license)

## Features

-   **Data Loading from CSV**: Reads training and scoring data directly from local CSV files.
-   **ID Column Handling**: Automatically identifies and removes specified identifier columns (`transaction_id`, `customer_id`, etc.) from datasets before training and prediction. These IDs are preserved and included in the API's output.
-   **Robust Data Preprocessing**:
    -   **Missing Value Imputation**: Uses K-Nearest Neighbors (KNN) imputation for numerical features and most-frequent imputation for categorical features.
    -   **Feature Scaling**: Applies `StandardScaler` to numerical features.
    -   **Dynamic Categorical Encoding**: Intelligently applies `OneHotEncoder` for low-cardinality categorical features (< 10 unique values) and `OrdinalEncoder` for high-cardinality features (>= 10 unique values). Both encoders handle unseen categories gracefully in new data.
-   **Comprehensive Class Imbalance Strategies**: Evaluates and compares different approaches:
    -   No sampling
    -   SMOTE (Synthetic Minority Oversampling Technique) only
    -   Random Undersampling only
    -   A combination of SMOTE and Random Undersampling
-   **Multiple Machine Learning Models**: Benchmarks the performance of:
    -   Logistic Regression
    -   Random Forest Classifier
    -   XGBoost Classifier (eXtreme Gradient Boosting)
-   **Hyperparameter Tuning**: Utilizes `GridSearchCV` with `StratifiedKFold` cross-validation to find optimal hyperparameters for each model and sampling strategy combination, optimizing for **F1-score**.
-   **XGBoost Class Weighting**: Incorporates tuning of `scale_pos_weight` for `XGBClassifier` to explicitly address class imbalance during its training.
-   **Best Model Selection**: Automatically identifies the overall best model and sampling strategy based on the highest F1-score achieved.
-   **Model Persistence**: Saves the entire trained model pipeline (including preprocessor and best estimator) as a `.pkl` file using `joblib`.
-   **Real-time Prediction API**: A Flask application served by Gunicorn provides a RESTful endpoint for low-latency predictions on new transaction data.
-   **Docker Support**: Includes a `Dockerfile` for easy containerization and consistent deployment across environments.

## Supported Python Version

This project is specifically designed and tested with **Python 3.12.4**. It is highly recommended to use this exact version for consistent local development and production deployments to avoid dependency conflicts.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/fraud-detection-model.git](https://github.com/your-username/fraud-detection-model.git)
    cd fraud-detection-model
    ```
2.  **Ensure Python 3.12.4 is Installed**:
    Verify your Python version:
    ```bash
    python --version
    # or
    python3.12 --version
    ```
    If not installed, download and install Python 3.12.4 from [python.org](https://www.python.org/downloads/). Ensure it's added to your system's PATH.

3.  **Create a Python 3.12.4 virtual environment** (recommended for managing dependencies):
    ```bash
    # On Windows (Command Prompt/PowerShell - use 'py -3.12' if 'python' isn't 3.12):
    py -3.12 -m venv .venv
    # On macOS/Linux (Bash/Zsh - use 'python3.12' if 'python' isn't 3.12):
    python3.12 -m venv .venv
    ```
    *This explicitly creates the virtual environment using your Python 3.12.4 executable.*

4.  **Activate the virtual environment**:
    ```bash
    # On Windows (Command Prompt/PowerShell):
    .\.venv\Scripts\activate
    # On macOS/Linux (Bash/Zsh):
    source ./.venv/bin/activate
    ```
    You should see `(.venv)` in your terminal prompt.

5.  **Install dependencies** from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

fraud_detection_project/
├── fraud_detection.py        # Main script for training, evaluation, batch prediction
├── api.py                    # Flask API for real-time predictions
├── sample_fraud_data.csv                # Your training data (e.g., 10k rows)
├── data_to_be_predicted.csv                 # Your unseen data for scoring/prediction (e.g., a few rows for testing)
├── requirements.txt          # Python package dependencies
├── Dockerfile                # Instructions for building the Docker image
├── .gitignore                # Specifies files/folders to ignore from Git
├── best_fraud_model.pkl      # Generated after the first successful training run
└── README.md                 # This documentation file

## Usage

### Data Preparation

1.  **`sample_fraud_data.csv`**:
    * Contains your training data.
    * **Must include all feature columns, your target variable** (e.g., `is_fraud`), and **any ID columns** (e.g., `transaction_id`, `customer_id`).
    * **Important**: Update `target_column_name` and `id_columns` variables in `fraud_detection.py` and `api.py` to precisely match your CSV headers.

    Example `sample_fraud_data.csv` structure:
    ```csv
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
    ```

2.  **`data_to_be_predicted.csv`**:
    * Contains unseen data for which you want predictions.
    * **Must have the same feature columns and ID columns as `sample_fraud_data.csv`, in the same order.**
    * **Must NOT contain the target column.**

    Example `data_to_be_predicted.csv` structure:
    ```csv
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
    ```
    **Place both `sample_fraud_data.csv` and `data_to_be_predicted.csv` in the root of your `fraud_detection_project` directory.**

### Step 1: Train and Save the Model

Run the main script to train the model, evaluate it, and save the best-performing pipeline. **You've already done this successfully!**

```bash
python fraud_detection.py