import sqlite3

import pandas as pd
import pytest

from data.generate import DROUGHT_MARKERS, N_LINES, N_MARKERS, generate

MARKER_COLS = [f"m{k}" for k in range(1, N_MARKERS + 1)]


def _read(db_path):
    con = sqlite3.connect(db_path)
    try:
        return {t: pd.read_sql(f"SELECT * FROM {t}", con) for t in ("lines", "environments", "trials")}
    finally:
        con.close()


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    return _read(generate(tmp_path_factory.mktemp("db") / "breeding.db"))


def test_row_counts_and_columns(tables):
    lines, envs, trials = tables["lines"], tables["environments"], tables["trials"]
    assert len(lines) == N_LINES
    assert len(envs) == 12
    assert [c for c in lines.columns if c not in ("line_id", "family", "maturity_group")] == MARKER_COLS
    assert set(lines[MARKER_COLS].to_numpy().ravel()) == {0, 1}
    assert {"yield_t_ha", "disease_score", "plant_height_cm"} <= set(trials.columns)


def test_each_line_in_four_to_six_environments(tables):
    counts = tables["trials"].groupby("line_id").size()
    assert len(counts) == N_LINES
    assert counts.between(4, 6).all()


def test_ranges_and_referential_integrity(tables):
    lines, envs, trials = tables["lines"], tables["environments"], tables["trials"]
    assert trials["disease_score"].between(1, 9).all()
    assert (trials["yield_t_ha"] > 0).all()
    assert set(envs["stress"]) == {"none", "drought", "heat"}
    assert set(trials["line_id"]) <= set(lines["line_id"])
    assert set(trials["env_id"]) <= set(envs["env_id"])
    assert "Cerrado-2027" in set(envs["env_id"])
    assert "L042" in set(lines["line_id"])


def test_deterministic_with_fixed_seed(tmp_path):
    a, b = _read(generate(tmp_path / "a.db")), _read(generate(tmp_path / "b.db"))
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_changes_data(tmp_path):
    a, b = _read(generate(tmp_path / "a.db", seed=1)), _read(generate(tmp_path / "b.db", seed=2))
    assert not a["trials"].equals(b["trials"])


def test_drought_markers_help_only_under_drought(tables):
    df = tables["trials"].merge(tables["lines"], on="line_id").merge(tables["environments"], on="env_id")
    high = df[[f"m{k}" for k in DROUGHT_MARKERS]].sum(axis=1) >= 3

    def gap(mask):
        return df[mask & high]["yield_t_ha"].mean() - df[mask & ~high]["yield_t_ha"].mean()

    assert gap(df["stress"] == "drought") > gap(df["stress"] != "drought") + 0.5
