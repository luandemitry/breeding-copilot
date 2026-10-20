import sqlite3

import pytest

from app.sql_guard import MAX_LIMIT, SQLGuardError, open_readonly, run_query, validate

VALID = [
    "SELECT line_id FROM lines LIMIT 5",
    "select line_id, family from lines limit 10;",
    "SELECT l.line_id, AVG(t.yield_t_ha) AS y FROM trials t JOIN lines l ON l.line_id = t.line_id "
    "GROUP BY l.line_id ORDER BY y DESC LIMIT 5",
    "SELECT t.line_id FROM trials t JOIN environments e ON e.env_id = t.env_id WHERE e.stress = 'drought' LIMIT 3",
    "WITH d AS (SELECT env_id FROM environments WHERE stress = 'drought') "
    "SELECT AVG(yield_t_ha) FROM trials WHERE env_id IN (SELECT env_id FROM d) LIMIT 1",
    "SELECT line_id FROM lines WHERE line_id IN (SELECT line_id FROM trials WHERE yield_t_ha > 6) LIMIT 5",
]

REJECTED = [
    ("DROP TABLE trials", "only SELECT"),
    ("DELETE FROM trials", "only SELECT"),
    ("UPDATE trials SET yield_t_ha = 99", "only SELECT"),
    ("INSERT INTO lines (line_id) VALUES ('X')", "only SELECT"),
    ("CREATE TABLE x (a INT)", "only SELECT"),
    ("ALTER TABLE lines ADD COLUMN z INT", "only SELECT"),
    ("PRAGMA table_info(lines)", "only SELECT"),
    ("ATTACH DATABASE 'x.db' AS x", "only SELECT"),
    ("SELECT line_id FROM lines LIMIT 5; DROP TABLE trials", "exactly one statement"),
    ("SELECT 1 LIMIT 1; SELECT 2 LIMIT 1", "exactly one statement"),
    ("SELECT * FROM lines", "LIMIT is required"),
    ("SELECT * FROM sqlite_master LIMIT 5", "table not allowed"),
    ("SELECT * FROM users LIMIT 5", "table not allowed"),
    ("SELECT * FROM main.trials LIMIT 5", "schema-qualified"),
    ("SELECT * FROM pragma_table_info('lines') LIMIT 5", "table functions"),
    ("SELECT line_id FROM lines WHERE line_id IN (SELECT name FROM sqlite_master) LIMIT 5", "table not allowed"),
    ("SELECT load_extension('evil') LIMIT 1", "forbidden function"),
    ("SELECT line_id FROM lines LIMIT (SELECT 5)", "LIMIT must be a plain integer"),
    ("", "empty"),
    ("   ", "empty"),
    ("SELEC nonsense ((", "SQL|SELECT"),
]


@pytest.mark.parametrize("sql", VALID)
def test_accepts_valid_selects(sql):
    assert validate(sql).upper().startswith(("SELECT", "WITH"))


@pytest.mark.parametrize(("sql", "reason"), REJECTED)
def test_rejects_unsafe_sql(sql, reason):
    with pytest.raises(SQLGuardError, match=reason):
        validate(sql)


def test_comments_are_stripped_from_executed_sql():
    safe = validate("SELECT line_id FROM lines /* ; DROP TABLE lines */ LIMIT 2 -- trailing")
    assert "DROP" not in safe.upper()
    assert "--" not in safe


def test_limit_is_clamped_to_max():
    assert f"LIMIT {MAX_LIMIT}" in validate("SELECT line_id FROM lines LIMIT 999999")
    assert "LIMIT 7" in validate("SELECT line_id FROM lines LIMIT 100", max_limit=7)


def test_run_query_returns_rows_and_columns(synthetic_db):
    sql, cols, rows, truncated = run_query(synthetic_db, "SELECT line_id, family FROM lines ORDER BY line_id LIMIT 3")
    assert cols == ["line_id", "family"]
    assert [r[0] for r in rows] == ["L001", "L002", "L003"]
    assert "LIMIT 3" in sql
    assert truncated is False  # reaching the query's own LIMIT is not truncation


def test_truncated_only_when_the_system_cap_cut_the_result(synthetic_db):
    _, _, rows, truncated = run_query(synthetic_db, "SELECT line_id FROM lines ORDER BY line_id LIMIT 999999", max_limit=7)
    assert len(rows) == 7 and truncated is True
    # A requested limit above the cap is clamped, but a small result is complete.
    _, _, rows, truncated = run_query(synthetic_db, "SELECT line_id FROM lines WHERE line_id = 'L042' LIMIT 999999", max_limit=7)
    assert rows == [["L042"]] and truncated is False
    # LIMIT equal to the cap is the query's own choice, not a clamp.
    _, _, rows, truncated = run_query(synthetic_db, "SELECT line_id FROM lines LIMIT 7", max_limit=7)
    assert len(rows) == 7 and truncated is False


def test_run_query_wraps_sqlite_errors(synthetic_db):
    with pytest.raises(SQLGuardError, match="query failed"):
        run_query(synthetic_db, "SELECT nope FROM lines LIMIT 1")


def test_connection_is_read_only_even_without_the_guard(synthetic_db):
    con = open_readonly(synthetic_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM trials")
        with pytest.raises(sqlite3.OperationalError):
            con.execute("CREATE TABLE evil (a INT)")
    finally:
        con.close()
    check = sqlite3.connect(synthetic_db)
    try:
        assert check.execute("SELECT COUNT(*) FROM trials").fetchone()[0] > 0
    finally:
        check.close()
