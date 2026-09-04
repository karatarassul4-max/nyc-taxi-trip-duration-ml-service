# Model Card: NYC Taxi Trip Duration ML Service

## Model Details

- Project: NYC Taxi Trip Duration ML Service
- Task type: regression
- Target: `trip_duration_minutes`
- Best current model: LightGBM regression pipeline
- Model artifact: generated locally at `artifacts/model.joblib`
- Training code: `src/taxi_duration/models/train.py`
- Inference API: `src/taxi_duration/api/main.py`

## Intended Use

The model estimates the duration of a NYC Yellow Taxi trip before the trip is completed. It is intended as a portfolio production ML service that demonstrates leakage-safe tabular ML, experiment tracking, API serving, Docker packaging, tests, CI, and basic drift monitoring.

This model is not intended for pricing, driver compensation, passenger eligibility, or any high-stakes operational decision.

## Data

Source: NYC Taxi & Limousine Commission Yellow Taxi Trip Record Data.

The default experiment uses:

- Train: January 2024
- Validation: February 2024
- Test: March 2024

The local development config caps each month to the first 120,000 raw rows before cleaning. A larger config is available at `configs/train_large.yaml` and uses the first 500,000 raw rows before cleaning.

## Features

The model uses only fields that can be known before the trip is completed:

- passenger count
- trip distance
- pickup hour
- pickup day of week
- pickup month
- weekend flag
- same pickup/dropoff zone flag
- pickup location ID
- dropoff location ID
- rate code ID
- payment type

Excluded leakage-prone fields include fare amount, tip amount, total amount, and dropoff timestamp.

## Evaluation

Primary metric: MAE in minutes.

The baseline predicts the median trip duration from the training split. Candidate models are compared on validation and test splits, with model selection based on validation MAE.

Current default-config result:

- Best model: LightGBM
- Test MAE: 3.2260 minutes
- Test RMSE: 5.2049 minutes
- Test R2: 0.8317

Larger-config result:

- Best model: LightGBM
- Test MAE: 3.4627 minutes
- Test RMSE: 5.7959 minutes
- Test R2: 0.7880

## Limitations

- The dataset is observational taxi trip data and reflects operational patterns, seasonality, holidays, weather, road conditions, and city events that are not fully represented in the current feature set.
- Location IDs are categorical zone identifiers, not exact GPS coordinates.
- The current service predicts a point estimate and does not return uncertainty intervals.
- Local development experiments use row caps for speed, so final production claims should be based on larger or full-month training runs.

## Monitoring

The project includes a drift report comparing the training split with the test split:

- PSI and KS test p-values for numeric features
- Total variation distance for categorical features

The default drift run shows expected temporal and location distribution shift between January and March 2024. In a real deployment, monitoring thresholds would trigger investigation or retraining rather than automatic model replacement.

## Ethical Considerations

The model should be used as an engineering demonstration and planning aid only. It should not be used to make decisions that materially affect drivers or passengers without additional validation, fairness analysis, uncertainty estimation, and operational review.
