# Disease-Risk-Prediction-MLOPS-Production-Ready-Project

# Disease Risk Prediction — FastAPI +  MLOps

This repository contains a MLOps-ready project to train a disease risk prediction model and serve it via a FastAPI app.

## Contents
- `src/train.py` — training script. Outputs model and supporting metadata into `models/`.
- `src/main.py` — FastAPI application that loads the trained pipeline and serves `/predict`.
- `Dockerfile` — containerize the app.
- `requirements.txt` — Python dependencies.
- `data/heart.csv` — example dataset location (not included).

## Quickstart (local)

1. Create a virtual environment and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

