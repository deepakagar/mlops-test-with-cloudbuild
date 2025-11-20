import pandas as pd
from sklearn.datasets import make_classification
from google.cloud import storage
import os
import argparse

def prepare_data(bucket_name, data_file_name):
    """Generates synthetic data and uploads it to GCS."""
    print("Starting data preparation...")

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    # Save data locally
    local_data_path = f'/tmp/{data_file_name}'
    df.to_csv(local_data_path, index=False)
    print(f"Synthetic data generated and saved locally to {local_data_path}")

    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(data_file_name)
    blob.upload_from_filename(local_data_path)
    print(f"Data uploaded to gs://{bucket_name}/{data_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation Script")
    parser.add_argument('--bucket_name', type=str, required=True, help='GCS bucket name to store data')
    parser.add_argument('--data_file_name', type=str, default='synthetic_data.csv', help='Name of the data file in GCS')
    args = parser.parse_args()

    prepare_data(args.bucket_name, args.data_file_name)
