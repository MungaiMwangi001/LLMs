from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
from pathlib import Path

# ----------------------
# Load model
# ----------------------
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "model.pkl"
model = joblib.load(model_path)

iris_classes = ["setosa", "versicolor", "virginica"]

# ----------------------
# Request Schemas
# ----------------------
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

class BatchInput(BaseModel):
    inputs: list[InputData]

# ----------------------
# FastAPI App
# ----------------------
app = FastAPI(
    title="🌸 Iris Classifier API",
    description="A FastAPI application that classifies iris flowers using ML",
    version="1.0.0",
)

# Enable CORS for frontend apps (React, Streamlit, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # update with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Endpoints
# ----------------------

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def home_page():
    """Simple HTML UI for user input"""
    html_content = """
    <html>
        <head>
            <title>Iris Classifier</title>
        </head>
        <body style="font-family: Arial; margin: 40px;">
            <h2>🌸 Iris Classifier</h2>
            <form action="/predict-form" method="post">
                <label>Feature 1: <input type="number" step="any" name="feature1" required></label><br><br>
                <label>Feature 2: <input type="number" step="any" name="feature2" required></label><br><br>
                <label>Feature 3: <input type="number" step="any" name="feature3" required></label><br><br>
                <label>Feature 4: <input type="number" step="any" name="feature4" required></label><br><br>
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict-form", response_class=HTMLResponse, tags=["UI"])
def predict_form(
    feature1: float = Form(...),
    feature2: float = Form(...),
    feature3: float = Form(...),
    feature4: float = Form(...)
):
    X = [[feature1, feature2, feature3, feature4]]
    prediction = model.predict(X)[0]
    flower_name = iris_classes[prediction]
    return f"<h3>Prediction: {flower_name}</h3>"

@app.get("/health", tags=["System"])
def health_check():
    """Check if API is alive"""
    return {"status": "ok"}

@app.get("/info", tags=["System"])
def get_info():
    """Model & dataset info"""
    return {
        "model": "Iris Classifier (sklearn)",
        "classes": iris_classes,
        "features": ["feature1", "feature2", "feature3", "feature4"],
    }

@app.post("/predict", tags=["ML"])
def predict(data: InputData):
    """Predict single input"""
    X = [[data.feature1, data.feature2, data.feature3, data.feature4]]
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    probs_dict = {iris_classes[i]: float(probabilities[i]) for i in range(len(iris_classes))}
    return {
        "prediction": int(prediction),
        "flower_name": iris_classes[prediction],
        "probabilities": probs_dict,
    }

@app.post("/predict-batch", tags=["ML"])
def predict_batch(batch: BatchInput):
    """Predict multiple inputs"""
    X = [[d.feature1, d.feature2, d.feature3, d.feature4] for d in batch.inputs]
    predictions = model.predict(X)
    probs = model.predict_proba(X)
    results = []
    for i, pred in enumerate(predictions):
        results.append({
            "prediction": int(pred),
            "flower_name": iris_classes[pred],
            "probabilities": {iris_classes[j]: float(probs[i][j]) for j in range(len(iris_classes))}
        })
    return {"results": results}
