import joblib
import pandas as pd

# Load model
MODEL_PATH = "models/iris_model.pkl"
model = joblib.load(MODEL_PATH)

# Define new test sample (adjust values as needed)
new_sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

# Predict
prediction = model.predict(new_sample)
print("🔍 Predicted Class:", prediction[0])
