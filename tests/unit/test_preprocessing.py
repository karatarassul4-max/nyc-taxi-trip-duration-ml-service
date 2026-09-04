from __future__ import annotations

import pandas as pd

from taxi_duration.data.preprocessing import clean_training_data

FEATURE_CONFIG = {
    "min_duration_minutes": 1,
    "max_duration_minutes": 120,
    "min_trip_distance": 0.1,
    "max_trip_distance": 60,
}


def test_clean_training_data_filters_invalid_rows_and_adds_target() -> None:
    frame = pd.DataFrame(
        [
            {
                "tpep_pickup_datetime": "2024-01-01 10:00:00",
                "tpep_dropoff_datetime": "2024-01-01 10:15:00",
                "passenger_count": 1,
                "trip_distance": 2.0,
                "PULocationID": 1,
                "DOLocationID": 2,
                "RatecodeID": 1,
                "payment_type": 1,
            },
            {
                "tpep_pickup_datetime": "2024-01-01 10:00:00",
                "tpep_dropoff_datetime": "2024-01-01 15:00:00",
                "passenger_count": 1,
                "trip_distance": 2.0,
                "PULocationID": 1,
                "DOLocationID": 2,
                "RatecodeID": 1,
                "payment_type": 1,
            },
        ]
    )

    clean = clean_training_data(frame, FEATURE_CONFIG)

    assert len(clean) == 1
    assert clean.loc[0, "trip_duration_minutes"] == 15
