import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/homl_heart_model.keras")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.pkl")
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "100"))
UNCERTAINTY_THRESHOLD = float(os.environ.get("UNCERTAINTY_THRESHOLD", "0.15"))

model = None
scaler = None


class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler

    if not Path(MODEL_PATH).exists() or not Path(SCALER_PATH).exists():
        raise RuntimeError(
            f"Model or scaler not found. Run `python -m src.train` first!\n"
            f"  Model path: {MODEL_PATH}\n  Scaler path: {SCALER_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={"MCDropout": MCDropout}
    )
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Heart Disease Prediction API",
    description="REST endpoint for heart disease risk prediction using a trained MLP.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientData(BaseModel):

    age: float = Field(..., ge=1, le=120, description="Patient age in years")
    sex: float = Field(..., ge=0, le=1, description="Sex (0 = Female, 1 = Male)")
    cp: float = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: float = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: float = Field(..., ge=0, le=4, description="Major vessels colored by fluoroscopy")
    thal: float = Field(..., ge=3, le=7, description="Thalassemia type")

    model_config = {"json_schema_extra": {
        "examples": [{
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 1, "ca": 0, "thal": 6,
        }]
    }}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict")
def predict_disease(data: PatientData):
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps,
        data.chol, data.fbs, data.restecg, data.thalach,
        data.exang, data.oldpeak, data.slope, data.ca, data.thal,
    ]])

    features_scaled = scaler.transform(features)

    try:
        X_repeated = np.repeat(features_scaled, MC_SAMPLES, axis=0)
        y_probas = model.predict(X_repeated, verbose=0, batch_size=256)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    mean_pred = float(y_probas.mean())
    std_pred = float(y_probas.std())

    return {
        "risk_probability": round(mean_pred, 4),
        "risk_percentage": round(mean_pred * 100, 2),
        "uncertainty_std": round(std_pred, 4),
        "requires_review": std_pred > UNCERTAINTY_THRESHOLD,
        "prediction": "Heart Disease Risk" if mean_pred >= 0.5 else "Healthy",
        "mc_samples": MC_SAMPLES,
        "message": "Prediction complete",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
