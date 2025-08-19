from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os 

# Load trained model
model = joblib.load("models/model.pkl")


# Class labels from Iris dataset
iris_classes = ["setosa", "versicolor", "virginica"]

# Define request body
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Create FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Iris Classifier API is running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    # Prepare input
    X = [[data.feature1, data.feature2, data.feature3, data.feature4]]

    # Prediction
    prediction = model.predict(X)[0]
    flower_name = iris_classes[prediction]

    # Probabilities
    probabilities = model.predict_proba(X)[0]  # array of [p_setosa, p_versicolor, p_virginica]
    probs_dict = {iris_classes[i]: float(probabilities[i]) for i in range(len(iris_classes))}

    return {
        "prediction": int(prediction),
        "flower_name": flower_name,
        "probabilities": probs_dict
    }



@app.get("/files")
def list_files():
    return {"files": os.listdir(".")}