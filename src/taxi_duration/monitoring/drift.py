from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from taxi_duration.config import ProjectConfig
from taxi_duration.data.preprocessing import clean_training_data, read_trip_data
from taxi_duration.features.build_features import TripFeatureBuilder
from taxi_duration.models.train import month_paths


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return float("nan")

    unique_values = pd.Index(expected.unique()).union(pd.Index(actual.unique()))
    if len(unique_values) <= bins:
        expected_pct = expected.value_counts(normalize=True).reindex(unique_values, fill_value=0)
        actual_pct = actual.value_counts(normalize=True).reindex(unique_values, fill_value=0)
        expected_pct = np.clip(expected_pct.to_numpy(), 1e-6, None)
        actual_pct = np.clip(actual_pct.to_numpy(), 1e-6, None)
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def categorical_shift(expected: pd.Series, actual: pd.Series, top_n: int = 10) -> float:
    expected_dist = expected.astype("string").value_counts(normalize=True).head(top_n)
    actual_dist = actual.astype("string").value_counts(normalize=True)
    categories = expected_dist.index.union(actual_dist.index)
    expected_aligned = expected_dist.reindex(categories, fill_value=0)
    actual_aligned = actual_dist.reindex(categories, fill_value=0)
    return float((expected_aligned - actual_aligned).abs().sum() / 2)


def build_drift_report(config: ProjectConfig) -> dict:
    max_rows = config.dataset.get("max_rows_per_month")
    reference_raw = read_trip_data(month_paths(config, "train"), max_rows)
    current_raw = read_trip_data(month_paths(config, "test"), max_rows)
    reference = TripFeatureBuilder().transform(clean_training_data(reference_raw, config.features))
    current = TripFeatureBuilder().transform(clean_training_data(current_raw, config.features))

    numeric = {}
    for column in config.features["numeric"]:
        numeric[column] = {
            "psi": population_stability_index(reference[column], current[column]),
            "ks_p_value": float(
                ks_2samp(reference[column].dropna(), current[column].dropna()).pvalue
            ),
        }

    categorical = {}
    for column in config.features["categorical"]:
        categorical[column] = {
            "total_variation_distance": categorical_shift(reference[column], current[column])
        }

    return {
        "reference_split": "train",
        "current_split": "test",
        "numeric": numeric,
        "categorical": categorical,
    }


def save_drift_report(
    report: dict,
    json_path: str | Path = "reports/drift_report.json",
    md_path: str | Path = "reports/drift_report.md",
) -> None:
    json_output = Path(json_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Drift Report",
        "",
        "Reference: train split. Current: test split.",
        "",
        "## Numeric Features",
        "",
        "| feature | PSI | KS p-value |",
        "| --- | ---: | ---: |",
    ]
    for feature, values in report["numeric"].items():
        lines.append(f"| {feature} | {values['psi']:.4f} | {values['ks_p_value']:.4g} |")
    lines.extend(
        [
            "",
            "## Categorical Features",
            "",
            "| feature | total variation distance |",
            "| --- | ---: |",
        ]
    )
    for feature, values in report["categorical"].items():
        lines.append(f"| {feature} | {values['total_variation_distance']:.4f} |")
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
