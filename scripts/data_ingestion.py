import pandas as pd
import os

# Define file path
DATA_PATH = "data/iris.data.csv"
PROCESSED_PATH = "data/processed_iris.csv"

def load_data():
    """Load raw data from CSV."""
    if not os.path.exists(DATA_PATH):
        print("Dataset not found!")
        return None
    return pd.read_csv(DATA_PATH)

def preprocess_data(df):
    """Basic preprocessing: Remove duplicates, handle missing values."""
    df = df.drop_duplicates()
    df = df.fillna(method="ffill")  # Forward fill missing values
    return df

def save_data(df):
    """Save preprocessed data."""
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved at {PROCESSED_PATH}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
        save_data(df)
