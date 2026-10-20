"""predict_performance: predicted yield (p50) with an 80% interval (p10-p90) per line in one environment."""

from __future__ import annotations

import os
from pathlib import Path

from app.config import ARTIFACT_DIR, DB_PATH
from app.schemas import Prediction
from models.bundle import MODEL_FILE, ModelBundle, load_bundle
from models.features import feature_matrix, load_tables

# An interval is "wide" when (p90 - p10) / p50 exceeds this ratio. On the default model about the
# widest 20% of predictions (mostly drought environments) cross 0.20.
DEFAULT_WIDE_RATIO = 0.20
MAX_LINES = 200

_bundle_cache: dict[tuple[str, float], ModelBundle] = {}


class UnknownIdError(ValueError):
    """A line or environment id does not exist in the database."""


def _cached_bundle(artifact_dir: Path | str) -> ModelBundle:
    path = Path(artifact_dir) / MODEL_FILE
    key = (str(path), path.stat().st_mtime if path.exists() else 0.0)
    if key not in _bundle_cache:
        _bundle_cache[key] = load_bundle(artifact_dir)
    return _bundle_cache[key]


def predict_performance(
    line_ids: list[str],
    env_id: str,
    *,
    bundle: ModelBundle | None = None,
    db_path: Path | str = DB_PATH,
    artifact_dir: Path | str = ARTIFACT_DIR,
    wide_ratio: float | None = None,
) -> list[Prediction]:
    """Predict yield (t/ha) for each line in ``env_id``.

    Raises :class:`UnknownIdError` for unknown ids and ``ModelNotTrainedError`` if ``make train`` was not run.
    """
    if not line_ids:
        raise ValueError("line_ids must not be empty")
    if len(line_ids) > MAX_LINES:
        raise ValueError(f"at most {MAX_LINES} lines per call")
    if wide_ratio is None:
        wide_ratio = float(os.getenv("PREDICT_WIDE_RATIO", DEFAULT_WIDE_RATIO))
    bundle = bundle or _cached_bundle(artifact_dir)

    lines, envs, _ = load_tables(db_path)
    env_rows = envs[envs["env_id"] == env_id]
    if env_rows.empty:
        raise UnknownIdError(f"unknown environment: {env_id}")
    unknown = sorted(set(line_ids) - set(lines["line_id"]))
    if unknown:
        raise UnknownIdError(f"unknown line(s): {', '.join(unknown)}")

    env = env_rows.iloc[0]
    frame = lines.set_index("line_id").loc[line_ids].reset_index()
    for col in ("rainfall_mm", "avg_temp_c", "stress"):
        frame[col] = env[col]
    p10, p50, p90 = bundle.predict(feature_matrix(frame))

    return [
        Prediction(
            line_id=line_id,
            env_id=env_id,
            p10=round(float(lo), 3),
            p50=round(float(mid), 3),
            p90=round(float(hi), 3),
            model_version=bundle.version,
            wide_interval=bool(mid > 0 and (hi - lo) / mid > wide_ratio),
        )
        for line_id, lo, mid, hi in zip(line_ids, p10, p50, p90)
    ]
