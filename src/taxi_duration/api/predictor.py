from __future__ import annotations

from pathlib import Path

import pandas as pd

from taxi_duration.api.schemas import TripPredictionRequest
from taxi_duration.models.registry import load_model


class TaxiDurationPredictor:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = load_model(self.model_path)

    def predict_one(self, request: TripPredictionRequest) -> float:
        row = pd.DataFrame(
            [
                {
                    "pickup_datetime": request.pickup_datetime,
                    "passenger_count": request.passenger_count,
                    "trip_distance": request.trip_distance,
                    "pickup_location_id": request.pickup_location_id,
                    "dropoff_location_id": request.dropoff_location_id,
                    "rate_code_id": request.rate_code_id,
                    "payment_type": request.payment_type,
                }
            ]
        )
        prediction = float(self.model.predict(row)[0])
        return max(0.0, prediction)
