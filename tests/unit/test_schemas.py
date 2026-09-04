from __future__ import annotations

import pytest
from pydantic import ValidationError

from taxi_duration.api.schemas import TripPredictionRequest


def test_prediction_request_accepts_valid_payload() -> None:
    request = TripPredictionRequest(
        pickup_datetime="2024-03-15T14:30:00",
        passenger_count=1,
        trip_distance=3.2,
        pickup_location_id=237,
        dropoff_location_id=161,
        rate_code_id=1,
        payment_type=1,
    )

    assert request.pickup_location_id == 237


def test_prediction_request_rejects_invalid_distance() -> None:
    with pytest.raises(ValidationError):
        TripPredictionRequest(
            pickup_datetime="2024-03-15T14:30:00",
            passenger_count=1,
            trip_distance=0,
            pickup_location_id=237,
            dropoff_location_id=161,
            rate_code_id=1,
            payment_type=1,
        )
