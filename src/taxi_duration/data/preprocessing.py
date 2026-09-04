from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
    "payment_type",
]


def read_trip_data(paths: list[str | Path], max_rows_per_month: int | None = None) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
        if max_rows_per_month:
            frame = frame.head(max_rows_per_month)
        frames.append(frame)
    if not frames:
        raise ValueError("No parquet files were provided.")
    return pd.concat(frames, ignore_index=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    pickup = pd.to_datetime(result["tpep_pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(result["tpep_dropoff_datetime"], errors="coerce")
    result["trip_duration_minutes"] = (dropoff - pickup).dt.total_seconds() / 60
    return result


def clean_training_data(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    result = add_target(df)
    result = result.dropna(subset=["trip_duration_minutes", "tpep_pickup_datetime"])

    min_duration = feature_config["min_duration_minutes"]
    max_duration = feature_config["max_duration_minutes"]
    min_distance = feature_config["min_trip_distance"]
    max_distance = feature_config["max_trip_distance"]

    result = result[
        result["trip_duration_minutes"].between(min_duration, max_duration)
        & result["trip_distance"].between(min_distance, max_distance)
        & result["passenger_count"].between(1, 6)
    ].copy()

    for column in ["PULocationID", "DOLocationID", "RatecodeID", "payment_type"]:
        result[column] = result[column].astype("Int64").astype("string")

    return result.reset_index(drop=True)
