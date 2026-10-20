"""Train the quantile yield models (p10 / p50 / p90) and report RMSE.

Three ``GradientBoostingRegressor(loss="quantile")`` models give a point estimate (median)
plus an 80% prediction interval. Artifacts are written to ``ARTIFACT_DIR`` (gitignored).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from app.config import ARTIFACT_DIR, DB_PATH
from models.bundle import METRICS_FILE, QUANTILES, ModelBundle, save_bundle
from models.features import TARGET, feature_matrix, training_frame

SEED = 42
TEST_SIZE = 0.2
PARAMS = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "min_samples_leaf": 5}


@dataclass
class Metrics:
    version: str
    n_train: int
    n_test: int
    rmse_p50: float
    rmse_baseline: float
    interval_coverage_p10_p90: float
    mean_interval_width: float


def train(
    db_path: Path | str = DB_PATH,
    artifact_dir: Path | str = ARTIFACT_DIR,
    seed: int = SEED,
    n_estimators: int | None = None,
) -> Metrics:
    df = training_frame(db_path)
    X, y = feature_matrix(df), df[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)

    params = {**PARAMS, **({"n_estimators": n_estimators} if n_estimators else {})}
    models = {
        q: GradientBoostingRegressor(loss="quantile", alpha=q, random_state=seed, **params).fit(X_tr, y_tr)
        for q in QUANTILES
    }
    # Version = content hash of data + hyperparameters + seed, so identical inputs give the same version.
    digest = hashlib.sha256(X.to_csv().encode() + json.dumps([params, seed], sort_keys=True).encode()).hexdigest()
    bundle = ModelBundle(version=f"qgbr-{digest[:8]}", models=models)

    p10, p50, p90 = bundle.predict(X_te)
    metrics = Metrics(
        version=bundle.version,
        n_train=len(X_tr),
        n_test=len(X_te),
        rmse_p50=float(np.sqrt(np.mean((y_te - p50) ** 2))),
        rmse_baseline=float(np.sqrt(np.mean((y_te - y_tr.mean()) ** 2))),
        interval_coverage_p10_p90=float(np.mean((y_te >= p10) & (y_te <= p90))),
        mean_interval_width=float(np.mean(p90 - p10)),
    )
    save_bundle(bundle, artifact_dir)
    (Path(artifact_dir) / METRICS_FILE).write_text(json.dumps(asdict(metrics), indent=2))
    return metrics


if __name__ == "__main__":
    m = train()
    print(f"Trained {m.version}: RMSE(p50)={m.rmse_p50:.3f} (baseline {m.rmse_baseline:.3f}), "
          f"p10-p90 coverage={m.interval_coverage_p10_p90:.0%}, mean width={m.mean_interval_width:.2f} t/ha")
