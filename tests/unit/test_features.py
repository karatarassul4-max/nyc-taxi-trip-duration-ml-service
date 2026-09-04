from __future__ import annotations

import pandas as pd

from taxi_duration.features.build_features import TripFeatureBuilder


def test_trip_feature_builder_creates_time_and_route_features() -> None:
    frame = pd.DataFrame(
        [
            {
                "pickup_datetime": "2024-01-06T08:30:00",
                "passenger_count": 1,
                "trip_distance": 2.5,
                "pickup_location_id": 161,
                "dropoff_location_id": 237,
                "rate_code_id": 1,
                "payment_type": 1,
            }
        ]
    )

    transformed = TripFeatureBuilder().transform(frame)

    assert transformed.loc[0, "pickup_hour"] == 8
    assert transformed.loc[0, "pickup_dayofweek"] == 5
    assert transformed.loc[0, "is_weekend"] == 1
    assert transformed.loc[0, "pickup_dropoff_same_zone"] == 0
    assert transformed.loc[0, "PULocationID"] == "161"
