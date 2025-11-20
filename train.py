import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from google.cloud import storage
import os
import argparse

def train_model(bucket_name, data_file_name, model_file_name):
    """Loads data from GCS, trains a model, and uploads the model to GCS."""
    print("Starting model training...")

    # Download data from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(data_file_name)
    local_data_path = f'/tmp/{data_file_name}'
    blob.download_to_filename(local_data_path)
    print(f"Data downloaded from gs://{bucket_name}/{data_file_name} to {local_data_path}")

    df = pd.read_csv(local_data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Save model locally
    local_model_path = f'/tmp/{model_file_name}'
    joblib.dump(model, local_model_path)
    print(f"Model saved locally to {local_model_path}")

    # Upload model to GCS
    blob = bucket.blob(model_file_name)
    blob.upload_from_filename(local_model_path)
    print(f"Model uploaded to gs://{bucket_name}/{model_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument('--bucket_name', type=str, required=True, help='GCS bucket name for data and model artifacts')
    parser.add_argument('--data_file_name', type=str, default='synthetic_data.csv', help='Name of the data file in GCS')
    parser.add_argument('--model_file_name', type=str, default='logistic_regression_model.joblib', help='Name of the trained model file in GCS')
    args = parser.parse_args()

    train_model(args.bucket_name, args.data_file_name, args.model_file_name)

