from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()  # This should be defined at the top level

# Load the trained model
MODEL_PATH = "models/iris_model.pkl"
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Welcome to the Iris Prediction API!"}

@app.post("/predict/")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                          columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    prediction = model.predict(sample)
    return {"Predicted Class": prediction[0]}
