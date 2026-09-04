from __future__ import annotations

import pandas as pd

from taxi_duration.config import ProjectConfig
from taxi_duration.models.train import train_models


def _write_month(path, day: int) -> None:
    rows = []
    for i in range(40):
        pickup = pd.Timestamp(year=2024, month=1, day=day, hour=i % 24, minute=0)
        duration = 5 + (i % 10) + (i % 3)
        rows.append(
            {
                "tpep_pickup_datetime": pickup,
                "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=duration),
                "passenger_count": 1 + (i % 3),
                "trip_distance": 0.5 + (i % 8),
                "PULocationID": 100 + (i % 4),
                "DOLocationID": 130 + (i % 5),
                "RatecodeID": 1,
                "payment_type": 1 + (i % 2),
            }
        )
    pd.DataFrame(rows).to_parquet(path)


def test_training_smoke(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_month(raw_dir / "yellow_tripdata_2024-01.parquet", 1)
    _write_month(raw_dir / "yellow_tripdata_2024-02.parquet", 2)
    _write_month(raw_dir / "yellow_tripdata_2024-03.parquet", 3)

    monkeypatch.chdir(tmp_path)
    config = ProjectConfig(
        path=tmp_path / "config.yaml",
        values={
            "dataset": {
                "year": 2024,
                "train_months": [1],
                "validation_months": [2],
                "test_months": [3],
                "raw_dir": str(raw_dir),
                "max_rows_per_month": None,
            },
            "training": {
                "experiment_name": "smoke",
                "target": "trip_duration_minutes",
                "artifact_path": str(tmp_path / "artifacts" / "model.joblib"),
                "metrics_path": str(tmp_path / "reports" / "metrics.json"),
                "experiment_table_path": str(tmp_path / "reports" / "experiment_table.md"),
            },
            "features": {
                "min_duration_minutes": 1,
                "max_duration_minutes": 120,
                "min_trip_distance": 0.1,
                "max_trip_distance": 60,
                "numeric": [
                    "passenger_count",
                    "trip_distance",
                    "pickup_hour",
                    "pickup_dayofweek",
                    "pickup_month",
                    "is_weekend",
                    "pickup_dropoff_same_zone",
                ],
                "categorical": ["PULocationID", "DOLocationID", "RatecodeID", "payment_type"],
            },
            "tuning": {},
        },
    )

    result = train_models(config, requested=["baseline", "ridge"])

    assert result["best_model"] in {"baseline", "ridge"}
    assert (tmp_path / "artifacts" / "model.joblib").exists()
    assert (tmp_path / "reports" / "experiment_table.md").exists()
