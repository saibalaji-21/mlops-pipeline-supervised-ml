import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Define file paths
RAW_DATA_PATH = "data/iris.data.csv"
PROCESSED_DATA_PATH = "data/processed_iris.csv"

def load_data():
    """Load raw dataset from CSV."""
    if not os.path.exists(RAW_DATA_PATH):
        print("Dataset not found!")
        return None
    return pd.read_csv(RAW_DATA_PATH)

def handle_missing_values(df):
    """Fill missing values with mean (if numeric) or mode (if categorical)."""
    for column in df.columns:
        if df[column].dtype == 'object':  # Categorical
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:  # Numeric
            df[column].fillna(df[column].mean(), inplace=True)
    return df

def encode_categorical(df):
    """Convert categorical labels into numerical values."""
    if 'species' in df.columns:  # Check if dataset has categorical labels
        le = LabelEncoder()
        df['species'] = le.fit_transform(df['species'])
    return df

def normalize_features(df):
    """Normalize numerical features using StandardScaler."""
    feature_columns = df.select_dtypes(include=['number']).columns  # Select only numerical columns
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

def save_preprocessed_data(df):
    """Save the preprocessed data."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Preprocessed data saved at {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df = handle_missing_values(df)
        df = encode_categorical(df)  # Encoding before scaling
        df = normalize_features(df)  # Scaling only numeric values
        save_preprocessed_data(df)
