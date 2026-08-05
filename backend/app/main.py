"""
main.py
-------
FastAPI application that serves the trained health-risk model.

Key flow for viva / report Question 5:
1. Angular UI collects student lifestyle inputs.
2. Browser sends POST /predict with JSON body.
3. FastAPI validates input using StudentHealthInput (schemas.py).
4. Input is converted to a one-row pandas DataFrame.
5. Trained sklearn Pipeline predicts class + probabilities.
6. JSON response is returned to Angular for display.
"""

from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import get_feature_columns, get_pipeline, load_model_bundle
from app.schemas import PredictionResponse, StudentHealthInput

app = FastAPI(
    title="Student Health Risk Predictor API",
    description="Predicts health_condition: at-risk | unhealthy | fit",
    version="1.0.0",
)

# Allow Angular (localhost:4200) to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

METRICS_PATH = Path(__file__).resolve().parents[1] / "models" / "metrics.json"


@app.on_event("startup")
def startup_event():
    """Load model when the server starts so first request is faster."""
    try:
        load_model_bundle()
        print("Model loaded successfully.")
    except FileNotFoundError as exc:
        print(f"WARNING: {exc}")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Student Health Risk Predictor API",
        "docs": "/docs",
    }


@app.get("/model-info")
def model_info():
    """Return model classes and training metrics if available."""
    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    metrics = None
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    return {
        "classes": bundle.get("classes"),
        "feature_columns": bundle.get("feature_columns"),
        "metrics": metrics,
        "model_type": "RandomForestClassifier inside sklearn Pipeline",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: StudentHealthInput):
    """
    Predict health_condition for one student.

    This is the main function invoked by the Angular Predict page.
    """
    try:
        pipeline = get_pipeline()
        feature_columns = get_feature_columns()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Convert validated input to a DataFrame with correct column order
    row = {col: getattr(payload, col) for col in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)

    try:
        pred = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0]
        classes = list(pipeline.classes_)
        probabilities = {
            cls: round(float(p), 4) for cls, p in zip(classes, proba)
        }
    except Exception as exc:  # noqa: BLE001 - return friendly API error
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictionResponse(
        health_condition=str(pred),
        probabilities=probabilities,
        message=f"Predicted student health condition: {pred}",
    )
