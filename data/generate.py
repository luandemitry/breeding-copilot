"""Generate the synthetic breeding-trials SQLite database (reproducible, fixed seed).

All data is synthetic and deliberately simplified:

    yield = base + genetic effect + environment effect + G x E interaction + noise

* genetic effect: weighted sum of a few additive markers
* environment effect: random environment offset plus a penalty for drought / heat stress
* G x E: a few "drought markers" only help in drought environments
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from app.config import DB_PATH

SEED = 42
N_LINES = 200
N_MARKERS = 50
N_FAMILIES = 8
BASE_YIELD = 4.0
MATURITY_GROUPS = ["early", "medium", "late"]

# Markers with an additive yield effect, drought-only (G x E) markers and disease-resistance markers.
YIELD_MARKERS = list(range(1, 11))
DROUGHT_MARKERS = list(range(11, 16))
RESISTANCE_MARKERS = list(range(16, 21))

STRESS_PENALTY = {"none": 0.0, "drought": -1.2, "heat": -0.8}

# (region, year) -> stress. Names double as env_id, e.g. "Cerrado-2027".
ENVIRONMENTS = {
    ("Cerrado", 2025): "none",
    ("Cerrado", 2026): "drought",
    ("Cerrado", 2027): "none",
    ("Matopiba", 2025): "drought",
    ("Matopiba", 2026): "heat",
    ("Matopiba", 2027): "drought",
    ("Sul", 2025): "none",
    ("Sul", 2026): "heat",
    ("Sul", 2027): "none",
    ("Pampa", 2025): "none",
    ("Pampa", 2026): "none",
    ("Pampa", 2027): "heat",
}

SCHEMA = """
CREATE TABLE lines (
    line_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    maturity_group TEXT NOT NULL,
    {markers}
);
CREATE TABLE environments (
    env_id TEXT PRIMARY KEY,
    region TEXT NOT NULL,
    year INTEGER NOT NULL,
    rainfall_mm REAL NOT NULL,
    avg_temp_c REAL NOT NULL,
    stress TEXT NOT NULL CHECK (stress IN ('none', 'drought', 'heat'))
);
CREATE TABLE trials (
    line_id TEXT NOT NULL REFERENCES lines(line_id),
    env_id TEXT NOT NULL REFERENCES environments(env_id),
    yield_t_ha REAL NOT NULL,
    disease_score INTEGER NOT NULL CHECK (disease_score BETWEEN 1 AND 9),
    plant_height_cm REAL NOT NULL,
    PRIMARY KEY (line_id, env_id)
);
"""


def _make_lines(rng: np.random.Generator):
    line_ids = [f"L{i:03d}" for i in range(1, N_LINES + 1)]
    families = [f"F{rng.integers(1, N_FAMILIES + 1)}" for _ in line_ids]
    maturity = [str(rng.choice(MATURITY_GROUPS)) for _ in line_ids]
    # Family-specific allele frequencies give related lines similar genotypes.
    freqs = {f: rng.beta(2, 2, size=N_MARKERS) for f in sorted(set(families))}
    markers = np.array([(rng.random(N_MARKERS) < freqs[f]).astype(int) for f in families])
    return line_ids, families, maturity, markers


def _make_environments(rng: np.random.Generator) -> list[dict]:
    envs = []
    for (region, year), stress in ENVIRONMENTS.items():
        if stress == "drought":
            rainfall = rng.uniform(300, 500)
        elif stress == "heat":
            rainfall = rng.uniform(600, 850)
        else:
            rainfall = rng.uniform(800, 1100)
        temp = rng.uniform(30, 33) if stress == "heat" else rng.uniform(21, 27)
        envs.append(
            {
                "env_id": f"{region}-{year}",
                "region": region,
                "year": year,
                "rainfall_mm": round(float(rainfall), 1),
                "avg_temp_c": round(float(temp), 1),
                "stress": stress,
                "offset": float(rng.normal(0, 0.4)),
            }
        )
    return envs


def generate(db_path: Path | str = DB_PATH, seed: int = SEED) -> Path:
    """Create the SQLite database at ``db_path`` (overwriting it) and return the path."""
    rng = np.random.default_rng(seed)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    line_ids, families, maturity, markers = _make_lines(rng)
    envs = _make_environments(rng)

    yield_w = rng.normal(0.30, 0.10, size=len(YIELD_MARKERS))
    drought_w = rng.normal(0.60, 0.10, size=len(DROUGHT_MARKERS))
    cols = lambda ms: [m - 1 for m in ms]  # noqa: E731  (marker numbers -> column indexes)

    maturity_height = {"early": 85.0, "medium": 100.0, "late": 115.0}
    trial_rows = []
    for i, line_id in enumerate(line_ids):
        genetic = float(markers[i, cols(YIELD_MARKERS)] @ yield_w)
        drought_genes = float(markers[i, cols(DROUGHT_MARKERS)] @ drought_w)
        resistance = int(markers[i, cols(RESISTANCE_MARKERS)].sum())
        n_trials = int(rng.integers(4, 7))  # 4-6 environments per line
        for j in rng.choice(len(envs), size=n_trials, replace=False):
            env = envs[j]
            gxe = drought_genes if env["stress"] == "drought" else 0.0
            noise = float(rng.normal(0, 0.3))
            y = BASE_YIELD + genetic + env["offset"] + STRESS_PENALTY[env["stress"]] + gxe + noise
            disease = 5 - 0.7 * resistance + (env["rainfall_mm"] - 800) / 400 + rng.normal(0, 0.8)
            height = maturity_height[maturity[i]] + rng.normal(0, 6)
            trial_rows.append(
                (
                    line_id,
                    env["env_id"],
                    round(max(0.5, y), 3),
                    int(np.clip(round(disease), 1, 9)),
                    round(float(height), 1),
                )
            )

    marker_cols = ", ".join(f"m{k} INTEGER NOT NULL CHECK (m{k} IN (0, 1))" for k in range(1, N_MARKERS + 1))
    line_rows = [
        (lid, fam, mat, *map(int, markers[i]))
        for i, (lid, fam, mat) in enumerate(zip(line_ids, families, maturity))
    ]
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.executescript(SCHEMA.format(markers=marker_cols))
            con.executemany(f"INSERT INTO lines VALUES (?, ?, ?, {', '.join('?' * N_MARKERS)})", line_rows)
            con.executemany(
                "INSERT INTO environments VALUES (?, ?, ?, ?, ?, ?)",
                [(e["env_id"], e["region"], e["year"], e["rainfall_mm"], e["avg_temp_c"], e["stress"]) for e in envs],
            )
            con.executemany("INSERT INTO trials VALUES (?, ?, ?, ?, ?)", trial_rows)
    finally:
        con.close()
    return db_path


if __name__ == "__main__":
    path = generate()
    print(f"Wrote synthetic database to {path}")
