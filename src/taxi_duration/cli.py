from __future__ import annotations

import argparse
import json
import logging

import joblib

from taxi_duration.config import load_config
from taxi_duration.data.download import download_months
from taxi_duration.logging import configure_logging
from taxi_duration.models.evaluate import regression_metrics, save_metrics
from taxi_duration.models.train import load_split, train_models
from taxi_duration.models.tune import tune_hist_gbr
from taxi_duration.monitoring.drift import build_drift_report, save_drift_report

LOGGER = logging.getLogger(__name__)


def download_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    configured_months = (
        config.dataset["train_months"]
        + config.dataset["validation_months"]
        + config.dataset["test_months"]
    )
    months = sorted(set(configured_months))
    download_months(config.dataset["year"], months, config.dataset["raw_dir"])


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = train_models(config, args.models)
    LOGGER.info(json.dumps(result, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    X_test, y_test = load_split(config, "test")
    model = joblib.load(args.model_path)
    metrics = regression_metrics(y_test, model.predict(X_test))
    save_metrics(metrics, "reports/evaluation_metrics.json")
    LOGGER.info(json.dumps(metrics, indent=2))


def tune(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = tune_hist_gbr(config)
    LOGGER.info(json.dumps(result, indent=2))


def monitor_drift(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    report = build_drift_report(config)
    save_drift_report(report)
    LOGGER.info("Saved drift report to reports/drift_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC taxi duration ML service CLI")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in [
        ("download-data", download_data),
        ("train", train),
        ("evaluate", evaluate),
        ("tune", tune),
        ("monitor-drift", monitor_drift),
    ]:
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(handler=handler)
        subparser.add_argument("--config", default="configs/train.yaml")
        if name == "train":
            subparser.add_argument(
                "--models",
                nargs="+",
                default=None,
                help="Models: baseline ridge hist_gbr lightgbm",
            )
        if name == "evaluate":
            subparser.add_argument("--model-path", default="artifacts/model.joblib")

    args = parser.parse_args()
    configure_logging(args.log_level)
    args.handler(args)


if __name__ == "__main__":
    main()
