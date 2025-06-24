import pandas as pd
import joblib
from flask import Flask, request, jsonify
import os
import sys

# --- Configuration ---
MODEL_FILE_NAME = 'best_fraud_model.pkl'
TRAIN_FILE_PATH = 'sample_fraud_data.csv' # Used to infer original column order for input validation
TARGET_COLUMN_NAME = 'is_fraud'
ID_COLUMNS = ['transaction_id', 'timestamp', 'currency', 'merchant_id', 'device_id',
                  'customer_id', 'last_login_time'] # Columns to ignore for model input


app = Flask(__name__)
model_pipeline = None # Will hold the loaded FraudDetectionModel instance


def load_model_and_prepare_for_api():
    """
    Loads the pre-trained FraudDetectionModel instance.
    This function also attempts to load a sample of the training data
    to infer the expected feature columns and their order for API input validation.
    """
    global model_pipeline

    if not os.path.exists(MODEL_FILE_NAME):
        print(f"Error: Model file '{MODEL_FILE_NAME}' not found.")
        print("Please train the model first by running 'python fraud_detection.py' to generate the model file.")
        sys.exit(1) # Exit if the model isn't available

    try:
        model_pipeline = joblib.load(MODEL_FILE_NAME)
        print("Model loaded successfully.")

        # Optional: Load a small sample of training data to get feature names and order
        # This is useful for robust API input validation
        if os.path.exists(TRAIN_FILE_PATH):
            try:
                sample_df = pd.read_csv(TRAIN_FILE_PATH, nrows=1) # Load just one row for column names
                # Exclude target and ID columns to get the expected feature columns for the model
                feature_columns = [col for col in sample_df.columns if col not in ID_COLUMNS and col != TARGET_COLUMN_NAME]
                model_pipeline.expected_features = feature_columns
                print(f"Expected features for prediction: {model_pipeline.expected_features}")
            except Exception as e:
                print(f"Warning: Could not read '{TRAIN_FILE_PATH}' to infer expected features: {e}")
                model_pipeline.expected_features = None # Indicates we can't reliably validate input columns
        else:
            print(f"Warning: '{TRAIN_FILE_PATH}' not found. Cannot infer expected feature columns for API validation.")
            model_pipeline.expected_features = None

    except Exception as e:
        print(f"Error loading the model file '{MODEL_FILE_NAME}': {e}")
        print("Please ensure the model file is valid. Exiting.")
        sys.exit(1)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for real-time fraud prediction.
    Expects a JSON array of transaction data.
    """
    if model_pipeline is None:
        return jsonify({"error": "Model not initialized. Server setup error."}), 500

    data = request.get_json(force=True) # Get JSON data from the request body

    if not isinstance(data, list):
        data = [data] # Ensure data is always a list of dictionaries for consistent processing

    try:
        input_df = pd.DataFrame(data)

        # Validate input columns if expected_features were inferred
        if model_pipeline.expected_features is not None:
            missing_cols = [col for col in model_pipeline.expected_features if col not in input_df.columns]
            if missing_cols:
                return jsonify({"error": f"Missing required features in input: {missing_cols}"}), 400
            # Ensure order of columns matches training, and select only expected features
            processed_input_df = input_df[model_pipeline.expected_features]
        else:
            # If expected_features couldn't be inferred, attempt to remove only known ID columns
            processed_input_df = input_df.copy()
            for col in ID_COLUMNS:
                if col in processed_input_df.columns:
                    processed_input_df = processed_input_df.drop(columns=[col])


        # Make predictions
        predictions = model_pipeline.predict(processed_input_df)

        # Prepare response including original IDs and predictions
        results = []
        for i, pred in enumerate(predictions):
            item = {}
            # Include original IDs if they exist in the input_df
            for id_col in ['transaction_id', 'timestamp', 'amount']:
                if id_col in input_df.columns:
                    item[id_col] = input_df.iloc[i][id_col]
            # Add all original features back to the result, or just the prediction
            # For simplicity, we'll just add the prediction and IDs
            item['predicted_is_fraud'] = int(pred) # Convert numpy.int64 to standard int for JSON
            results.append(item)

        return jsonify(results)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e), "message": "An error occurred during prediction. Check server logs."}), 400

# This block runs when api.py is executed (e.g., by gunicorn or python api.py)
if __name__ == '__main__':
    load_model_and_prepare_for_api()
    app.run(host='0.0.0.0', port=8000, debug=False) # debug=False for production