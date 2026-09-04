from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:  # noqa: ANN001
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def experiment_table(results: list[dict]) -> str:
    lines = [
        "| model | split | MAE | RMSE | R2 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['split']} | {row['mae']:.4f} | "
            f"{row['rmse']:.4f} | {row['r2']:.4f} |"
        )
    return "\n".join(lines) + "\n"
