from __future__ import annotations

from pathlib import Path

import joblib


def load_model(path: str | Path):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `make train` before serving the API."
        )
    return joblib.load(model_path)
