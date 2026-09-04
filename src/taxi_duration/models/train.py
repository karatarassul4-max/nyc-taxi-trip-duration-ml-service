from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from taxi_duration.config import ProjectConfig
from taxi_duration.data.preprocessing import clean_training_data, read_trip_data
from taxi_duration.features.build_features import TripFeatureBuilder
from taxi_duration.models.baseline import MedianDurationRegressor
from taxi_duration.models.evaluate import experiment_table, regression_metrics, save_metrics

LOGGER = logging.getLogger(__name__)


def month_paths(config: ProjectConfig, split: str) -> list[Path]:
    year = config.dataset["year"]
    months = config.dataset[f"{split}_months"]
    raw_dir = Path(config.dataset["raw_dir"])
    return [raw_dir / f"yellow_tripdata_{year}-{month:02d}.parquet" for month in months]


def load_split(config: ProjectConfig, split: str) -> tuple[pd.DataFrame, pd.Series]:
    df = read_trip_data(month_paths(config, split), config.dataset.get("max_rows_per_month"))
    clean = clean_training_data(df, config.features)
    y = clean[config.training["target"]]
    leakage_columns = ["tpep_dropoff_datetime", config.training["target"]]
    X = clean.drop(columns=[column for column in leakage_columns if column in clean.columns])
    return X, y


def build_preprocessor(config: ProjectConfig, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), config.features["numeric"]),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=20,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                config.features["categorical"],
            ),
        ],
        remainder="drop",
    )


def model_candidates(
    config: ProjectConfig,
    requested: list[str] | None = None,
) -> dict[str, Pipeline]:
    requested_set = set(requested or ["baseline", "ridge", "hist_gbr"])
    candidates: dict[str, Pipeline] = {}

    if "baseline" in requested_set:
        candidates["baseline"] = Pipeline(
            [("features", TripFeatureBuilder()), ("model", MedianDurationRegressor())]
        )

    if "ridge" in requested_set:
        candidates["ridge"] = Pipeline(
            [
                ("features", TripFeatureBuilder()),
                ("preprocess", build_preprocessor(config, scale_numeric=True)),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    if "random_forest" in requested_set:
        candidates["random_forest"] = Pipeline(
            [
                ("features", TripFeatureBuilder()),
                ("preprocess", build_preprocessor(config)),
                ("model", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)),
            ]
        )

    if "hist_gbr" in requested_set:
        candidates["hist_gbr"] = Pipeline(
            [
                ("features", TripFeatureBuilder()),
                ("preprocess", build_preprocessor(config)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=180,
                        learning_rate=0.08,
                        random_state=42,
                    ),
                ),
            ]
        )

    if "lightgbm" in requested_set:
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            LOGGER.warning("LightGBM is not installed; skipping lightgbm candidate.")
        else:
            candidates["lightgbm"] = Pipeline(
                [
                    ("features", TripFeatureBuilder()),
                    ("preprocess", build_preprocessor(config)),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=500,
                            learning_rate=0.05,
                            num_leaves=64,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

    return candidates


def train_models(config: ProjectConfig, requested: list[str] | None = None) -> dict:
    X_train, y_train = load_split(config, "train")
    X_val, y_val = load_split(config, "validation")
    X_test, y_test = load_split(config, "test")

    mlflow.set_experiment(config.training["experiment_name"])
    all_results = []
    best = {"name": None, "metric": float("inf"), "pipeline": None}

    for name, pipeline in model_candidates(config, requested).items():
        LOGGER.info("Training %s", name)
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            mlflow.log_param("model", name)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("validation_rows", len(X_val))
            mlflow.log_param("test_rows", len(X_test))

            for split_name, X_split, y_split in [
                ("validation", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                metrics = regression_metrics(y_split, pipeline.predict(X_split))
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{split_name}_{metric_name}", value)
                all_results.append({"model": name, "split": split_name, **metrics})

            validation_mae = next(
                row["mae"]
                for row in all_results
                if row["model"] == name and row["split"] == "validation"
            )
            if validation_mae < best["metric"]:
                best = {"name": name, "metric": validation_mae, "pipeline": pipeline}

    if best["pipeline"] is None:
        raise RuntimeError("No model candidates were trained.")

    artifact_path = Path(config.training["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], artifact_path)
    LOGGER.info("Saved best model '%s' to %s", best["name"], artifact_path)

    save_metrics(
        {"best_model": best["name"], "results": all_results},
        config.training["metrics_path"],
    )
    table = experiment_table(all_results)
    table_path = Path(config.training["experiment_table_path"])
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table, encoding="utf-8")
    return {"best_model": best["name"], "results": all_results}
