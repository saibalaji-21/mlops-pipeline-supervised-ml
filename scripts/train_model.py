import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

# Define file paths
PROCESSED_DATA_PATH = "data/processed_iris.csv"
MODEL_PATH = "models/iris_model.pkl"

def load_preprocessed_data():
    """Load preprocessed dataset and check column names."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("Preprocessed dataset not found!")
        return None
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print("✅ Loaded data columns:", df.columns)  # Debugging Step
    return df

def split_data(df):
    """Split data into training and testing sets."""
    target_column = "species" if "species" in df.columns else df.columns[-1]  # Auto-detect target
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target label
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def save_model(model):
    """Save trained model."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    df = load_preprocessed_data()
    if df is not None:
        X_train, X_test, y_train, y_test = split_data(df)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model)
