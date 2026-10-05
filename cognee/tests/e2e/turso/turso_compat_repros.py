"""Minimal reproductions of the Turso engine findings listed in docs/turso-local.md.

Each finding is a self-contained function against the installed ``pyturso`` (and,
for the dialect findings, its bundled SQLAlchemy dialects); none touches cognee.
Run it to see which findings still hold, e.g. after a ``pyturso`` upgrade:

    python cognee/tests/e2e/turso/turso_compat_repros.py

Every line reads ``PRESENT`` (the engine still behaves as documented, so cognee's
workaround is still needed) or ``FIXED`` (the workaround can likely be retired).
Not a pytest module: the file name keeps it out of collection on purpose, since
a fixed finding is good news, not a failure.
"""

from __future__ import annotations

import asyncio
import datetime
import importlib.metadata
import os
import sys
import tempfile
from collections.abc import Callable

import sqlalchemy.exc
import turso

# What a reproduction may raise: the engine's DB-API errors, SQLAlchemy's wrappers,
# and the AttributeError of the broken bundled async dialect (finding 1).
ENGINE_ERRORS = (turso.Error, sqlalchemy.exc.SQLAlchemyError, AttributeError)


def _connect(directory: str, name: str = "repro.db"):
    return turso.connect(os.path.join(directory, name))


def _fails(run: Callable[[], object], *fragments: str) -> tuple[bool, str]:
    """(True, message) when ``run`` raises an error naming one of ``fragments``."""
    try:
        result = run()
    except ENGINE_ERRORS as error:
        message = str(error).splitlines()[0]
        if any(fragment.lower() in message.lower() for fragment in fragments):
            return True, message
        return False, f"unexpected error: {message}"
    return False, f"no error (result: {result!r})"


def finding_1_aioturso_dialect_has_stop(directory: str) -> tuple[bool, str]:
    """The bundled ``sqlite+aioturso`` dialect fails on SQLAlchemy 2.0.4x+."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    async def probe():
        engine = create_async_engine(f"sqlite+aioturso:///{directory}/aio.db")
        try:
            async with engine.connect() as connection:
                return (await connection.execute(text("SELECT 1"))).scalar()
        finally:
            await engine.dispose()

    return _fails(lambda: asyncio.run(probe()), "has_stop")


def finding_2_dialect_reflection_returns_nothing(directory: str) -> tuple[bool, str]:
    """The bundled dialect reflects no indexes although ``PRAGMA index_list`` has them."""
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine(f"sqlite+turso:///{directory}/reflect.db")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)"))
            connection.execute(text("CREATE INDEX ix_t_v ON t (v)"))
        with engine.connect() as connection:
            pragma = connection.execute(text("PRAGMA index_list('t')")).all()
            reflected = inspect(connection).get_indexes("t")
    finally:
        engine.dispose()
    if pragma and not reflected:
        return True, f"PRAGMA index_list has {len(pragma)} index(es), get_indexes() returns []"
    return False, f"get_indexes() returns {reflected!r}"


def finding_3_subquery_in_upsert_set(directory: str) -> tuple[bool, str]:
    """A scalar subquery inside ``ON CONFLICT DO UPDATE SET`` is rejected."""
    connection = _connect(directory)
    connection.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    connection.execute("CREATE TABLE src (v TEXT)")
    connection.execute("INSERT INTO src VALUES ('x')")
    connection.commit()
    return _fails(
        lambda: connection.execute(
            "INSERT INTO t (id, v) VALUES (1, 'a') "
            "ON CONFLICT (id) DO UPDATE SET v = (SELECT v FROM src LIMIT 1)"
        ).fetchall(),
        "subquery is not supported",
    )


def finding_4_bind_inside_nested_json_each(directory: str) -> tuple[bool, str]:
    """A bind inside a nested ``json_each(?)`` subquery is misapplied.

    The tag-removal UPDATE the vector adapter used to run in SQL. Stock SQLite
    strips the tag; the engine either errors or silently matches no rows.
    """
    import sqlite3

    statement = (
        "UPDATE t SET payload = json_set(payload, '$.belongs_to_set', ("
        "SELECT json_group_array(value) FROM json_each(payload, '$.belongs_to_set') "
        "WHERE value NOT IN (SELECT value FROM json_each(?)))) WHERE id IN (?, ?)"
    )

    def run(connection):
        connection.execute("CREATE TABLE t (id TEXT PRIMARY KEY, payload TEXT)")
        connection.execute(
            'INSERT INTO t VALUES (\'1\', \'{"belongs_to_set": ["a", "b"]}\'), '
            "('2', '{\"belongs_to_set\": [\"a\"]}')"
        )
        connection.commit()
        connection.execute(statement, ('["a"]', "1", "2")).fetchall()
        connection.commit()
        return connection.execute("SELECT id, payload FROM t ORDER BY id").fetchall()

    expected = run(sqlite3.connect(os.path.join(directory, "stock.db")))
    try:
        actual = run(_connect(directory))
    except ENGINE_ERRORS as error:
        return True, str(error).splitlines()[0]
    if actual != expected:
        return True, f"rows {actual!r}, stock SQLite gives {expected!r}"
    return False, "same rows as stock SQLite"


def finding_5_recursive_cte(directory: str) -> tuple[bool, str]:
    """``WITH RECURSIVE`` is not supported (0.7.x)."""
    connection = _connect(directory)
    return _fails(
        lambda: connection.execute(
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 3) "
            "SELECT count(*) FROM n"
        ).fetchall(),
        "recursive",
    )


def finding_6_quoted_identifiers_lowercased(directory: str) -> tuple[bool, str]:
    """A quoted mixed-case table name comes back lowercased from ``sqlite_master``."""
    connection = _connect(directory)
    connection.execute('CREATE TABLE "DocumentChunk_text" (id TEXT)')
    connection.commit()
    names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND lower(name) = ?",
            ("documentchunk_text",),
        ).fetchall()
    ]
    if names == ["documentchunk_text"]:
        return True, "stored as 'documentchunk_text'"
    return False, f"stored as {names!r}"


def finding_7_primitive_binds_only(directory: str) -> tuple[bool, str]:
    """Binding a ``datetime`` fails: only None, numbers, str and bytes are accepted."""
    connection = _connect(directory)
    connection.execute("CREATE TABLE t (at TEXT)")
    stamp = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    try:
        connection.execute("INSERT INTO t VALUES (?)", (stamp,))
        connection.commit()
    except ENGINE_ERRORS as error:
        return True, str(error).splitlines()[0]
    return False, "datetime bind accepted"


def finding_8_parenthesized_join(directory: str) -> tuple[bool, str]:
    """``JOIN (a JOIN b ON ...)`` in a FROM clause is rejected."""
    connection = _connect(directory)
    connection.execute("CREATE TABLE a (id INTEGER, b_id INTEGER)")
    connection.execute("CREATE TABLE b (id INTEGER)")
    connection.execute("CREATE TABLE c (id INTEGER)")
    connection.commit()
    return _fails(
        lambda: connection.execute(
            "SELECT a.id FROM a JOIN (b JOIN c ON b.id = c.id) ON a.b_id = b.id"
        ).fetchall(),
        "parenthesized",
    )


FINDINGS = [
    finding_1_aioturso_dialect_has_stop,
    finding_2_dialect_reflection_returns_nothing,
    finding_3_subquery_in_upsert_set,
    finding_4_bind_inside_nested_json_each,
    finding_5_recursive_cte,
    finding_6_quoted_identifiers_lowercased,
    finding_7_primitive_binds_only,
    finding_8_parenthesized_join,
]


def main() -> int:
    print(f"pyturso {importlib.metadata.version('pyturso')}")
    for finding in FINDINGS:
        with tempfile.TemporaryDirectory() as directory:
            try:
                present, detail = finding(directory)
            except ENGINE_ERRORS as error:  # a repro that breaks is reported, not hidden
                present, detail = None, f"repro failed: {str(error).splitlines()[0]}"
        status = {True: "PRESENT", False: "FIXED", None: "ERROR"}[present]
        print(f"{status:8} {finding.__name__}: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
