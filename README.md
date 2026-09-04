# NYC Taxi Trip Duration ML Service

Production-style machine learning service for predicting NYC Yellow Taxi trip duration before a trip is completed.

This project is designed as a portfolio-grade Junior ML Engineer project. It covers the full lifecycle from raw public data to experiment tracking, model artifacts, FastAPI inference, Docker, tests, CI, and drift monitoring.

## Problem

Predict `trip_duration_minutes` for a Yellow Taxi ride using only features that are available at pickup time.

The target is calculated from:

```text
tpep_dropoff_datetime - tpep_pickup_datetime
```

To avoid target leakage, fare, tip, total amount, and dropoff timestamp are not used as model inputs.

## Dataset

Source: [NYC Taxi & Limousine Commission Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

The pipeline downloads monthly Yellow Taxi parquet files from the TLC public data bucket. The default configuration uses:

- Train: January 2024
- Validation: February 2024
- Test: March 2024

The data is large, so `configs/train.yaml` limits rows per month by default for local development. Increase `max_rows_per_month` for a stronger final experiment.

## Architecture

```text
raw TLC parquet
  -> cleaning and leakage-safe target creation
  -> time/location/distance feature engineering
  -> baseline and model comparison
  -> MLflow experiment tracking
  -> saved sklearn pipeline artifact
  -> FastAPI inference API
  -> Docker image
  -> pytest and GitHub Actions CI
  -> drift report between train and test months
```

## Repository Structure

```text
configs/                 Training configuration
data/                    Local raw and processed data, ignored by git
notebooks/               EDA notebook
src/taxi_duration/       Production Python package
tests/                   Unit and integration tests
reports/                 Generated metrics and drift reports, ignored by git
artifacts/               Saved model artifacts, ignored by git
.github/workflows/       CI pipeline
```

## Models

Implemented candidates:

- `baseline`: median duration regressor
- `ridge`: linear baseline with preprocessing
- `hist_gbr`: sklearn histogram gradient boosting
- `lightgbm`: optional LightGBM model if the extra dependency is installed

Hyperparameter tuning uses a small `RandomizedSearchCV` search over meaningful `HistGradientBoostingRegressor` parameters.

## Metrics

Regression metrics:

- MAE
- RMSE
- R2

MAE is the primary model selection metric because it is directly interpretable in minutes.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
make install-dev
```

On Windows without `make`, run the commands from the `Makefile` directly.

## Run Pipeline

```bash
make download-data
make train-all
make tune
make monitor-drift
make test
```

Equivalent CLI:

```bash
taxi-duration download-data --config configs/train.yaml
taxi-duration train --config configs/train.yaml --models baseline ridge hist_gbr lightgbm
taxi-duration tune --config configs/train.yaml
taxi-duration monitor-drift --config configs/train.yaml
```

## Serve API

```bash
make serve
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"pickup_datetime\":\"2024-03-15T14:30:00\",\"passenger_count\":1,\"trip_distance\":3.2,\"pickup_location_id\":237,\"dropoff_location_id\":161,\"rate_code_id\":1,\"payment_type\":1}"
```

## Docker

```bash
make docker-build
make docker-run
```

Train locally first so `artifacts/model.joblib` exists.

## Experiment Results

Experiments below were run locally on September 4, 2026 using the default config:

- Train: January 2024, first 120,000 rows before cleaning
- Validation: February 2024, first 120,000 rows before cleaning
- Test: March 2024, first 120,000 rows before cleaning

Best model by validation MAE: `lightgbm`.

| model | split | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| baseline | validation | 8.1436 | 12.8789 | -0.0895 |
| baseline | test | 8.1280 | 13.1383 | -0.0725 |
| ridge | validation | 5.9753 | 7.6024 | 0.6204 |
| ridge | test | 6.3804 | 7.8618 | 0.6160 |
| hist_gbr | validation | 3.6671 | 5.8576 | 0.7746 |
| hist_gbr | test | 3.2968 | 5.3180 | 0.8243 |
| lightgbm | validation | 3.5874 | 5.7353 | 0.7839 |
| lightgbm | test | 3.2260 | 5.2049 | 0.8317 |

The tuned `HistGradientBoostingRegressor` reached validation MAE `3.6138` and test MAE `3.2432`, so the default saved model remains LightGBM.

Generated local files:

- `reports/metrics.json`
- `reports/experiment_table.md`
- `reports/tuning_metrics.json`
- local MLflow runs under `mlruns/`

## Drift Monitoring

The drift job compares the training split against the test split and writes:

- `reports/drift_report.json`
- `reports/drift_report.md`

It reports PSI and KS test p-values for numeric features, plus total variation distance for categorical features.

The first real drift run showed notable month/day-of-week and location distribution shift between the January train split and March test split. This is expected with temporal data and is useful for discussing monitoring thresholds and retraining cadence.

## Engineering Decisions

- Temporal train/validation/test split to better match production model evaluation.
- Leakage-safe features only.
- Baseline model included before advanced models.
- sklearn `Pipeline` stores preprocessing and model together.
- MLflow logs params and metrics for reproducibility.
- FastAPI/Pydantic boundary separates API payloads from training column names.
- Drift monitoring is intentionally lightweight and locally runnable.
