import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the original dataset
file_path = "data/processed_iris.csv"
df = pd.read_csv(file_path, header=None)  # No header in transformed data

# Define correct column names
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df.columns = column_names  # Assign correct column names

# Standardize numeric features
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])  # Scale only numerical columns

# Save the cleaned dataset
df.to_csv("data/processed_iris.csv", index=False)
print("✅ Fixed and saved processed dataset.")
