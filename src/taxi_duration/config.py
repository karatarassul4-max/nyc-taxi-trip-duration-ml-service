from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    values: dict[str, Any]
    path: Path

    @property
    def dataset(self) -> dict[str, Any]:
        return self.values["dataset"]

    @property
    def training(self) -> dict[str, Any]:
        return self.values["training"]

    @property
    def features(self) -> dict[str, Any]:
        return self.values["features"]

    @property
    def tuning(self) -> dict[str, Any]:
        return self.values.get("tuning", {})


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file)
    return ProjectConfig(values=values, path=config_path)
