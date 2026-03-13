from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from contextlib import asynccontextmanager

# Global variable to hold our model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model on startup
    try:
        ml_models["rf_model"] = joblib.load("../model/disease_risk_model.joblib")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(title="Disease Risk Prediction API", lifespan=lifespan)

# Define the expected input payload using Pydantic
class PatientData(BaseModel):
    age: float = Field(..., description="Patient's age")
    cholesterol: float = Field(..., description="Serum cholesterol level")
    resting_bp: float = Field(..., description="Resting blood pressure")
    max_heart_rate: float = Field(..., description="Maximum heart rate achieved")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Disease Risk Prediction API. Send a POST request to /predict."}

@app.post("/predict")
def predict_risk(data: PatientData):
    if "rf_model" not in ml_models:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Convert incoming data to a DataFrame
    input_df = pd.DataFrame([data.model_dump()])
    
    # Make prediction and get probability
    model = ml_models["rf_model"]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] # Probability of the positive class
    
    risk_status = "High Risk" if prediction == 1 else "Low Risk"
    
    return {
        "risk_status": risk_status,
        "risk_probability": round(float(probability), 4)
    }