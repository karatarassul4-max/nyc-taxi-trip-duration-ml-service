from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class TripPredictionRequest(BaseModel):
    pickup_datetime: datetime
    passenger_count: int = Field(ge=1, le=6)
    trip_distance: float = Field(gt=0, le=100)
    pickup_location_id: int = Field(ge=1, le=265)
    dropoff_location_id: int = Field(ge=1, le=265)
    rate_code_id: int = Field(default=1, ge=1)
    payment_type: int = Field(default=1, ge=1)

    @field_validator("trip_distance")
    @classmethod
    def realistic_distance(cls, value: float) -> float:
        if value < 0.01:
            raise ValueError("trip_distance must be positive")
        return value


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float
    model_path: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
