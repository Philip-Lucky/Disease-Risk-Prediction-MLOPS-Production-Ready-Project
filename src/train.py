#!/usr/bin/env python3
"""
Train a disease-risk model and save a pipeline that includes preprocessing.
Produces:
 - models/model_pipeline.joblib
 - models/features.json
 - models/metrics.json
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    # Use XGBoost if installed for better tabular performance
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


def train(args):
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(args.data_path)
    if args.target_col not in df.columns:
        raise SystemExit(f"Target column '{args.target_col}' not found in dataset")

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    feature_names = list(X.columns)
    print(f"[train] Features ({len(feature_names)}): {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if args.stratify else None
    )

    clf = None
    if args.use_xgboost and HAS_XGB:
        clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=args.random_state, n_estimators=100)
        print("[train] Using XGBoost classifier")
    else:
        clf = RandomForestClassifier(n_estimators=150, random_state=args.random_state)
        print("[train] Using RandomForest classifier")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        # fallback for estimators without predict_proba
        y_prob = np.clip(pipeline.predict(X_test), 0, 1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(set(y_test)) > 1 else None,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "xgboost" if (args.use_xgboost and HAS_XGB) else "random_forest",
    }

    model_path = os.path.join("models", args.output_name)
    joblib.dump(pipeline, model_path)
    print(f"[train] Saved pipeline to {model_path}")

    features_path = os.path.join("models", "features.json")
    with open(features_path, "w") as f:
        json.dump({"features": feature_names}, f)
    print(f"[train] Saved features list to {features_path}")

    metrics_path = os.path.join("models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Saved metrics to {metrics_path}")

    print("[train] Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train disease risk model")
    p.add_argument("--data-path", default="data/heart.csv", help="CSV dataset path (features + target column)")
    p.add_argument("--target-col", default="target", help="Name of the target column in the CSV")
    p.add_argument("--output-name", default="model_pipeline.joblib", help="Output model filename under models/")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--use-xgboost", action="store_true", help="If installed, use XGBoost instead of RandomForest")
    p.add_argument("--stratify", action="store_true", help="Stratify train/test split by target")
    args = p.parse_args()
    train(args)