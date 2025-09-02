from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
from pathlib import Path

# Path to the model
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "model.pkl"

# Load model
model = joblib.load(model_path)
iris_classes = ["setosa", "versicolor", "virginica"]

# FastAPI app
app = FastAPI()

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    feature1: float = Form(...),
    feature2: float = Form(...),
    feature3: float = Form(...),
    feature4: float = Form(...),
):
    X = [[feature1, feature2, feature3, feature4]]
    prediction = model.predict(X)[0]
    flower_name = iris_classes[prediction]
    probabilities = model.predict_proba(X)[0]
    probs_dict = {iris_classes[i]: round(float(probabilities[i]), 3) for i in range(len(iris_classes))}
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": flower_name, "probs": probs_dict}
    )


@app.post("/predict")
def predict(data: InputData):
    X = [[data.feature1, data.feature2, data.feature3, data.feature4]]
    prediction = model.predict(X)[0]
    flower_name = iris_classes[prediction]
    probabilities = model.predict_proba(X)[0]
    probs_dict = {iris_classes[i]: float(probabilities[i]) for i in range(len(iris_classes))}

    return {
        "prediction": int(prediction),
        "flower_name": flower_name,
        "probabilities": probs_dict
    }
