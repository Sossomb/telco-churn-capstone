"""
API REST — prédiction du churn client.
Lancer avec : uvicorn api.main:app --reload (depuis la racine du projet)
"""
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

app = FastAPI(title="API Prédiction Churn Telco", version="1.0.0")

pipeline = joblib.load(MODEL_DIR / "pipeline.joblib")
with open(MODEL_DIR / "model_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

DECISION_THRESHOLD = metadata["decision_threshold"]


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok", "version": metadata["model_version"]}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        X = pd.DataFrame([customer.model_dump()])
        proba = pipeline.predict_proba(X)[0, 1]
        prediction = int(proba >= DECISION_THRESHOLD)
        return {
            "churn_prediction": prediction,
            "churn_probability": round(float(proba), 4),
            "decision_threshold": DECISION_THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model-info")
def model_info():
    return {
        "expected_features": metadata["expected_features"],
        "model_version": metadata["model_version"],
        "decision_threshold": metadata["decision_threshold"],
        "validation_performance": metadata["validation_performance"],
    }