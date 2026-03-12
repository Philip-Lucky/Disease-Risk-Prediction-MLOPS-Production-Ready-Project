#!/usr/bin/env python3
"""
FastAPI app for serving the trained model.

POST /predict
  - Accepts a JSON object of feature_name -> numeric value
  - Returns risk probability and risk label

GET /health
  - Basic health & model info
"""

import json
import os
from typing import Dict, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI(title="Disease Risk Prediction API")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_pipeline.joblib")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "models/features.json")
METRICS_PATH = os.environ.get("METRICS_PATH", "models/metrics.json")


class FeaturePayload(BaseModel):
    __root__: Dict[str, float]


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            features = json.load(f).get("features", [])
    else:
        # If features.json not present, attempt to infer (not recommended)
        try:
            # sklearn pipeline -> last step estimator feature names not always available
            features = []
        except Exception:
            features = []

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    return model, features, metrics


model, FEATURES_ORDER, MODEL_METRICS = load_artifacts()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "trained_at": MODEL_METRICS.get("trained_at"),
        "model_type": MODEL_METRICS.get("model_type"),
        "n_features_expected": len(FEATURES_ORDER),
    }


@app.post("/predict")
async def predict(payload: FeaturePayload):
    data: Dict[str, Any] = payload.__root__

    # validate features
    if not FEATURES_ORDER:
        raise HTTPException(status_code=500, detail="Server misconfigured: features list missing. Re-train with train.py.")

    missing = [f for f in FEATURES_ORDER if f not in data]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}. Expected features: {FEATURES_ORDER}",
        )

    # Build ordered input
    x = np.array([[float(data[f]) for f in FEATURES_ORDER]])
    try:
        prob = float(model.predict_proba(x)[0, 1])
    except Exception:
        # fallback to predict output if no predict_proba
        pred = model.predict(x)[0]
        prob = float(pred)

    risk_label = "high" if prob >= 0.5 else "low"

    return {
        "risk_score": prob,
        "risk": risk_label,
        "model_type": MODEL_METRICS.get("model_type"),
        "trained_at": MODEL_METRICS.get("trained_at"),
    }


@app.post("/predict/raw")
async def predict_raw(request: Request):
    """
    Accept raw ordered list of numeric features as JSON list for quick testing:
    POST body example: [63, 1, 145, 233, ...]
    (Order must match features.json)
    """
    body = await request.json()
    if not isinstance(body, list):
        raise HTTPException(status_code=400, detail="Expected JSON list of numeric features")
    if len(body) != len(FEATURES_ORDER):
        raise HTTPException(status_code=400, detail=f"Expected {len(FEATURES_ORDER)} features in order: {FEATURES_ORDER}")
    x = np.array([list(map(float, body))])
    try:
        prob = float(model.predict_proba(x)[0, 1])
    except Exception:
        prob = float(model.predict(x)[0])
    return {"risk_score": prob, "risk": "high" if prob >= 0.5 else "low"}