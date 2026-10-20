"""Validate LLM-generated SQL on its AST (sqlglot) and run it read-only.

Rules enforced by :func:`validate`:

* exactly one statement, and it is a ``SELECT`` (CTEs allowed)
* only the three allowed tables (CTE aliases excepted), no schema qualifiers or table functions
* a ``LIMIT`` is mandatory and clamped to ``MAX_LIMIT``
* no DDL/DML/PRAGMA/ATTACH nodes and no file/extension functions anywhere in the tree

The *re-generated* SQL (not the original text) is what gets executed, so comments and other
tricks in the input never reach SQLite. The connection is additionally opened read-only.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

ALLOWED_TABLES = frozenset({"lines", "environments", "trials"})
MAX_LIMIT = 5000
QUERY_TIMEOUT_S = 5.0

_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Merge,
    exp.Command, exp.Pragma, exp.Attach, exp.Detach, exp.Set, exp.Transaction, exp.Commit,
    exp.Rollback, exp.Use, exp.Copy, exp.Into,
)  # fmt: skip
_FORBIDDEN_FUNCTIONS = frozenset(
    {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zeroblob", "randomblob"}
)


class SQLGuardError(ValueError):
    """The SQL was rejected by the guard (never executed)."""


def validate(sql: str, max_limit: int = MAX_LIMIT) -> str:
    """Return a safe, normalised (SQLite dialect) version of ``sql`` or raise :class:`SQLGuardError`."""
    if not sql or not sql.strip():
        raise SQLGuardError("empty SQL")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except SqlglotError as e:
        raise SQLGuardError(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise SQLGuardError("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Select):
        raise SQLGuardError("only SELECT statements are allowed")

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLGuardError(f"forbidden operation: {type(node).__name__}")
        if isinstance(node, exp.Anonymous) and str(node.name).lower() in _FORBIDDEN_FUNCTIONS:
            raise SQLGuardError(f"forbidden function: {node.name}")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SQLGuardError("table functions are not allowed")
        if table.args.get("db") or table.args.get("catalog"):
            raise SQLGuardError("schema-qualified tables are not allowed")
        name = table.name.lower()
        if name not in ALLOWED_TABLES and name not in cte_names:
            raise SQLGuardError(f"table not allowed: {table.name}")

    limit = tree.args.get("limit")
    if limit is None:
        raise SQLGuardError("LIMIT is required")
    value = limit.expression
    if not (isinstance(value, exp.Literal) and not value.is_string and value.name.isdigit()):
        raise SQLGuardError("LIMIT must be a plain integer")
    if int(value.name) > max_limit:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_limit)))

    return tree.sql(dialect="sqlite", comments=False)


def open_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open the SQLite database in read-only mode (writes fail at the connection level)."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def run_query(
    db_path: Path | str, sql: str, max_limit: int = MAX_LIMIT
) -> tuple[str, list[str], list[list], bool]:
    """Validate and execute ``sql`` read-only.

    Returns ``(executed_sql, columns, rows, truncated)``. ``truncated`` is True when the result
    hit the LIMIT, i.e. there may be more rows.
    """
    safe_sql = validate(sql, max_limit)
    limit = int(sqlglot.parse_one(safe_sql, read="sqlite").args["limit"].expression.name)
    con = open_readonly(db_path)
    deadline = time.monotonic() + QUERY_TIMEOUT_S
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(safe_sql)
        columns = [d[0] for d in cur.description]
        rows = [list(r) for r in cur.fetchall()]
    except sqlite3.Error as e:
        raise SQLGuardError(f"query failed: {e}") from e
    finally:
        con.close()
    return safe_sql, columns, rows, len(rows) >= limit
