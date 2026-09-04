from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from taxi_duration.api.predictor import TaxiDurationPredictor
from taxi_duration.api.schemas import HealthResponse, PredictionResponse, TripPredictionRequest
from taxi_duration.logging import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
predictor: TaxiDurationPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global predictor
    try:
        predictor = TaxiDurationPredictor(MODEL_PATH)
        LOGGER.info("Loaded model from %s", MODEL_PATH)
    except FileNotFoundError as exc:
        LOGGER.warning("%s", exc)
        predictor = None
    yield


app = FastAPI(
    title="NYC Taxi Trip Duration ML Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=predictor is not None)


@app.get("/model-info")
def model_info() -> dict[str, str | bool]:
    return {"model_path": MODEL_PATH, "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TripPredictionRequest) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run training first.")
    return PredictionResponse(
        predicted_duration_minutes=predictor.predict_one(request),
        model_path=str(predictor.model_path),
    )
