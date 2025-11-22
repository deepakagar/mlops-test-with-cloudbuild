import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from google.cloud import storage
import os
import argparse

def evaluate_model(bucket_name, data_file_name, model_file_name):
    """Loads data and model from GCS, evaluates the model, and prints metrics."""
    print("Starting model evaluation...")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download data from GCS
    blob = bucket.blob(data_file_name)
    local_data_path = f'/tmp/{data_file_name}'
    blob.download_to_filename(local_data_path)
    print(f"Data downloaded from gs://{bucket_name}/{data_file_name}")

    df = pd.read_csv(local_data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Download model from GCS
    blob = bucket.blob(model_file_name)
    local_model_path = f'/tmp/{model_file_name}'
    blob.download_to_filename(local_model_path)
    print(f"Model downloaded from gs://{bucket_name}/{model_file_name}")

    model = joblib.load(local_model_path)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")

    # In a real MLOps pipeline, you might push these metrics to a monitoring system
    # or decide whether to promote the model based on these metrics.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument('--bucket_name', type=str, required=True, help='GCS bucket name for data and model artifacts')
    parser.add_argument('--data_file_name', type=str, default='synthetic_data.csv', help='Name of the data file in GCS')
    parser.add_argument('--model_file_name', type=str, default='logistic_regression_model.joblib', help='Name of the trained model file in GCS')
    args = parser.parse_args()

    evaluate_model(args.bucket_name, args.data_file_name, args.model_file_name)

