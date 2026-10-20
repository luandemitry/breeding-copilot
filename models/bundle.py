"""Loading and using the trained quantile model bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from app.config import ARTIFACT_DIR
from models.features import FEATURE_COLUMNS

MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"
QUANTILES = (0.1, 0.5, 0.9)


class ModelNotTrainedError(FileNotFoundError):
    """Raised when no trained model artifact exists (run ``make train``)."""


@dataclass
class ModelBundle:
    version: str
    models: dict[float, GradientBoostingRegressor]

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p10, p50, p90); quantiles are sorted per row so they never cross."""
        raw = np.column_stack([self.models[q].predict(X[FEATURE_COLUMNS]) for q in QUANTILES])
        raw.sort(axis=1)
        return raw[:, 0], raw[:, 1], raw[:, 2]


def save_bundle(bundle: ModelBundle, artifact_dir: Path | str = ARTIFACT_DIR) -> Path:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / MODEL_FILE
    joblib.dump({"version": bundle.version, "models": bundle.models}, path)
    return path


def load_bundle(artifact_dir: Path | str = ARTIFACT_DIR) -> ModelBundle:
    path = Path(artifact_dir) / MODEL_FILE
    if not path.exists():
        raise ModelNotTrainedError(f"No trained model at {path}. Run `make train` first.")
    payload = joblib.load(path)
    return ModelBundle(version=payload["version"], models=payload["models"])
