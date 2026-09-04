# Senior ML Engineering Review

## Strengths

- The project uses a realistic public dataset instead of a toy benchmark.
- The train/validation/test split is temporal, which better matches production forecasting conditions.
- The pipeline avoids clear leakage fields such as fare, tip, total amount, and dropoff timestamp.
- Baseline, linear, sklearn boosting, and LightGBM models are compared with real metrics.
- The saved artifact contains preprocessing and model logic together as a sklearn pipeline.
- FastAPI, Pydantic schemas, Docker, tests, CI, logging, MLflow, and drift monitoring are included without overcomplicating the architecture.

## Risks And Follow-Ups

- Row-capped experiments are useful for local iteration. The repository includes both a fast 120k rows/month run and a larger 500k rows/month run, but full-month experiments would still be useful before making stronger production claims.
- The current features do not include weather, traffic, holidays, or zone-level borough metadata, so residual error will remain for unusual city conditions.
- Location IDs are high-cardinality categorical variables. One-hot encoding is simple and explainable, but target encoding or native categorical support could be tested in a future iteration.
- The API returns a point estimate only. A production version should consider prediction intervals or error bands.
- The monitoring job is batch-oriented and simulated against historical splits. A deployed service would need persisted request logs and scheduled monitoring.

## Verdict

This is a strong Junior ML Engineer portfolio project because it demonstrates practical production ML habits: real data, leakage control, reproducible experiments, model comparison, artifact packaging, API serving, CI, tests, and monitoring. The design is intentionally modest and runnable, which is exactly the right instinct for a second portfolio project that complements a more complex CV/VLM system.
