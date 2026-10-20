"""Feature engineering shared by training and the predict tool."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

MARKER_COLS = [f"m{k}" for k in range(1, 51)]
ENV_NUMERIC = ["rainfall_mm", "avg_temp_c"]
STRESS_LEVELS = ["none", "drought", "heat"]
FEATURE_COLUMNS = MARKER_COLS + ENV_NUMERIC + [f"stress_{s}" for s in STRESS_LEVELS]
TARGET = "yield_t_ha"


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model input from a frame holding marker columns plus environment columns."""
    out = df[MARKER_COLS + ENV_NUMERIC].astype(float).copy()
    for level in STRESS_LEVELS:
        out[f"stress_{level}"] = (df["stress"] == level).astype(float)
    return out[FEATURE_COLUMNS]


def load_tables(db_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the (lines, environments, trials) tables from the SQLite database."""
    con = sqlite3.connect(db_path)
    try:
        return tuple(  # type: ignore[return-value]
            pd.read_sql(f"SELECT * FROM {t}", con) for t in ("lines", "environments", "trials")
        )
    finally:
        con.close()


def training_frame(db_path: Path | str) -> pd.DataFrame:
    """One row per trial with line markers and environment covariates joined in."""
    lines, envs, trials = load_tables(db_path)
    return trials.merge(lines, on="line_id").merge(envs, on="env_id")
