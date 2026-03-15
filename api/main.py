"""
main.py  —  FastAPI Credit Risk Prediction API
------------------------------------------------
Endpoints:
  POST /predict          — predict default probability + risk tier + top SHAP factors
  GET  /health           — health check
  GET  /model-info       — model metadata and best metrics

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Causal ML API",
    description=(
        "Predicts loan default probability using causal ML. "
        "Returns probability, risk tier, and top 3 SHAP causal factors."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load artifacts at startup ─────────────────────────────────────────────────

MODEL_DIR = os.getenv("MODEL_DIR", "models")

model = None
feature_names = None
shap_artifacts = None
best_metrics = {}


@app.on_event("startup")
def load_artifacts():
    global model, feature_names, shap_artifacts, best_metrics
    try:
        model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
        feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
        if os.path.exists(f"{MODEL_DIR}/shap_artifacts.pkl"):
            shap_artifacts = joblib.load(f"{MODEL_DIR}/shap_artifacts.pkl")
        if os.path.exists(f"{MODEL_DIR}/best_metrics.csv"):
            best_metrics = pd.read_csv(f"{MODEL_DIR}/best_metrics.csv").to_dict("records")[0]
        print("✅ Artifacts loaded successfully.")
    except Exception as e:
        print(f"⚠️  Warning: Could not load artifacts: {e}")


# ── Schemas ───────────────────────────────────────────────────────────────────

class LoanApplication(BaseModel):
    loan_amnt:          float = Field(..., example=15000, description="Loan amount requested ($)")
    funded_amnt:        float = Field(..., example=15000)
    term:               int   = Field(..., example=36,    description="Loan term in months (36 or 60)")
    int_rate:           float = Field(..., example=13.5,  description="Interest rate (%)")
    installment:        float = Field(..., example=507.0)
    emp_length:         float = Field(..., example=5.0,   description="Employment length in years")
    home_ownership:     int   = Field(..., example=1,     description="0=RENT, 1=OWN, 2=MORTGAGE")
    annual_inc:         float = Field(..., example=60000, description="Annual income ($)")
    verification_status:int   = Field(..., example=1)
    purpose:            int   = Field(..., example=2)
    dti:                float = Field(..., example=18.5,  description="Debt-to-income ratio")
    delinq_2yrs:        float = Field(..., example=0.0)
    fico_score:         float = Field(..., example=700.0, description="FICO credit score")
    inq_last_6mths:     float = Field(..., example=1.0)
    open_acc:           float = Field(..., example=10.0)
    pub_rec:            float = Field(..., example=0.0)
    revol_bal:          float = Field(..., example=12000.0)
    revol_util:         float = Field(..., example=45.0)
    total_acc:          float = Field(..., example=20.0)
    initial_list_status:int   = Field(..., example=1)
    application_type:   int   = Field(..., example=0)
    grade_num:          int   = Field(..., example=3,     description="Grade (1=A to 7=G)")
    payment_to_income:  float = Field(..., example=0.10)
    loan_to_income:     float = Field(..., example=0.25)
    credit_utilization: float = Field(..., example=0.20)


class PredictionResponse(BaseModel):
    default_probability: float
    risk_tier:           str
    risk_score:          int
    top_3_causal_factors: list
    interpretation:      str


# ── Risk tier logic ───────────────────────────────────────────────────────────

def get_risk_tier(prob: float) -> tuple[str, int]:
    if prob < 0.20:
        return "LOW", int(prob * 500)
    elif prob < 0.45:
        return "MEDIUM", int(200 + prob * 500)
    elif prob < 0.65:
        return "HIGH", int(400 + prob * 400)
    else:
        return "VERY HIGH", int(600 + prob * 400)


# ── SHAP local explanation ────────────────────────────────────────────────────

def get_shap_factors(input_df: pd.DataFrame, n: int = 3) -> list[dict]:
    """Compute SHAP values for one applicant and return top-n factors."""
    if shap_artifacts is None:
        return [{"feature": "SHAP unavailable", "shap_value": 0.0, "direction": "unknown"}]
    try:
        explainer = shap_artifacts["explainer"]
        sv = explainer.shap_values(input_df)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv[0]
        sorted_idx = np.argsort(np.abs(sv))[::-1][:n]
        return [
            {
                "feature":     feature_names[i],
                "shap_value":  round(float(sv[i]), 4),
                "direction":   "increases default risk" if sv[i] > 0 else "decreases default risk",
            }
            for i in sorted_idx
        ]
    except Exception as e:
        return [{"error": str(e)}]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "shap_loaded":  shap_artifacts is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__ if model else "Not loaded",
        "features":   feature_names or [],
        "metrics":    best_metrics,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Build input DataFrame in correct column order
    data = application.dict()
    input_df = pd.DataFrame([data])

    # Align to training feature order
    if feature_names:
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[feature_names]

    # Predict
    prob = float(model.predict_proba(input_df)[0][1])
    risk_tier, risk_score = get_risk_tier(prob)

    # Causal SHAP factors
    top_factors = get_shap_factors(input_df)

    interpretation = (
        f"This applicant has a {prob:.1%} probability of default. "
        f"Risk is classified as {risk_tier}. "
        f"The primary driver is '{top_factors[0]['feature']}' "
        f"which {top_factors[0]['direction']}."
        if top_factors and "error" not in top_factors[0]
        else f"Default probability: {prob:.1%}. Risk: {risk_tier}."
    )

    return PredictionResponse(
        default_probability=round(prob, 4),
        risk_tier=risk_tier,
        risk_score=min(risk_score, 999),
        top_3_causal_factors=top_factors,
        interpretation=interpretation,
    )


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
