from __future__ import annotations

import logging
from pathlib import Path

import requests

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"


def download_yellow_taxi_month(year: int, month: int, raw_dir: str | Path) -> Path:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    destination = raw_path / f"yellow_tripdata_{year}-{month:02d}.parquet"
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("Dataset already exists: %s", destination)
        return destination

    url = BASE_URL.format(year=year, month=month)
    LOGGER.info("Downloading %s", url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    LOGGER.info("Saved %s (%.1f MB)", destination, destination.stat().st_size / 1024 / 1024)
    return destination


def download_months(year: int, months: list[int], raw_dir: str | Path) -> list[Path]:
    return [download_yellow_taxi_month(year, month, raw_dir) for month in months]
