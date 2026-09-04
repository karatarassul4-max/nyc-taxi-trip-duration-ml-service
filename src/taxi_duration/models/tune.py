from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from taxi_duration.config import ProjectConfig
from taxi_duration.features.build_features import TripFeatureBuilder
from taxi_duration.models.evaluate import regression_metrics, save_metrics
from taxi_duration.models.train import build_preprocessor, load_split

LOGGER = logging.getLogger(__name__)


def tune_hist_gbr(config: ProjectConfig) -> dict:
    X_train, y_train = load_split(config, "train")
    X_val, y_val = load_split(config, "validation")

    pipeline = Pipeline(
        [
            ("features", TripFeatureBuilder()),
            ("preprocess", build_preprocessor(config)),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=config.tuning.get("random_seed", 42),
                ),
            ),
        ]
    )
    param_distributions = {
        "model__learning_rate": uniform(0.03, 0.12),
        "model__max_iter": randint(100, 350),
        "model__max_leaf_nodes": randint(16, 64),
        "model__l2_regularization": uniform(0.0, 0.4),
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=config.tuning.get("n_iter", 12),
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=config.tuning.get("cv_splits", 3)),
        random_state=config.tuning.get("random_seed", 42),
        n_jobs=-1,
        verbose=1,
    )

    mlflow.set_experiment(config.training["experiment_name"])
    with mlflow.start_run(run_name="hist_gbr_tuned"):
        search.fit(X_train, y_train)
        metrics = regression_metrics(y_val, search.predict(X_val))
        mlflow.log_params(search.best_params_)
        for key, value in metrics.items():
            mlflow.log_metric(f"validation_{key}", value)

    tuned_path = Path("artifacts/model_tuned.joblib")
    tuned_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, tuned_path)
    result = {
        "best_params": search.best_params_,
        "validation": metrics,
        "artifact_path": str(tuned_path),
    }
    save_metrics(result, "reports/tuning_metrics.json")
    LOGGER.info("Saved tuned model to %s", tuned_path)
    return result
