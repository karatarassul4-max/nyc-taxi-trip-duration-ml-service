# CV Bullets

- Built an end-to-end production ML service for NYC taxi trip duration prediction using Pandas, scikit-learn, LightGBM, MLflow, FastAPI, Docker, pytest, and GitHub Actions.
- Designed leakage-safe preprocessing and temporal train/validation/test splits on real NYC TLC parquet data, improving test MAE from `8.13` min baseline to `3.23` min with LightGBM.
- Implemented reproducible experiment tracking, model artifact persistence, inference schemas, API health/model endpoints, and integration tests.
- Added lightweight drift monitoring with PSI, KS tests, and categorical distribution shift reports for production-style model observability.
