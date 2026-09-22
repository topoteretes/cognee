"""Minimal reproductions of the Turso engine gaps cognee's adapters work around.

Not a pytest module (no ``test_`` prefix): run it directly to print each finding's
status against the installed ``pyturso``::

    python cognee/tests/e2e/turso/turso_compat_repros.py

Each reproduction is a few statements on an in-memory database. "GAP" means the
engine still rejects the construct (cognee keeps its workaround), "OK" means the
upstream fix has landed. See ``docs/turso-local.md`` for the workarounds.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import os
import tempfile
from collections.abc import Callable


def _fresh():
    import turso

    connection = turso.connect(":memory:")
    connection.execute("CREATE TABLE v (id TEXT PRIMARY KEY, payload TEXT, vector F32_BLOB(2))")
    connection.execute("CREATE TABLE e (source_id TEXT, target_id TEXT)")
    connection.execute(
        "INSERT INTO v VALUES ('1', '{\"belongs_to_set\":[\"A\"]}', vector32('[1,0]'))"
    )
    return connection


def recursive_cte(connection) -> None:
    connection.execute(
        "WITH RECURSIVE n(id, hops) AS (SELECT '1', 0 UNION ALL "
        "SELECT e.target_id, n.hops + 1 FROM n JOIN e ON e.source_id = n.id WHERE n.hops < 2) "
        "SELECT count(*) FROM n"
    ).fetchall()


def scalar_subquery_in_upsert_set(connection) -> None:
    connection.execute(
        "INSERT INTO v VALUES ('1', '{\"belongs_to_set\":[\"B\"]}', vector32('[0,1]')) "
        "ON CONFLICT(id) DO UPDATE SET payload = json_set(excluded.payload, '$.belongs_to_set', "
        "(SELECT json_group_array(value) FROM ("
        "SELECT value FROM json_each(json_extract(v.payload, '$.belongs_to_set')) UNION "
        "SELECT value FROM json_each(json_extract(excluded.payload, '$.belongs_to_set')))))"
    )


def bind_inside_nested_json_each(connection) -> None:
    connection.execute(
        "UPDATE v SET payload = json_set(payload, '$.belongs_to_set', "
        "(SELECT json_group_array(value) FROM json_each(payload, '$.belongs_to_set') "
        "WHERE value NOT IN (SELECT value FROM json_each(?)))) WHERE id = '1'",
        ('["A"]',),
    )


def datetime_bind_parameter(connection) -> None:
    connection.execute("SELECT ?", (datetime.datetime.now(datetime.timezone.utc),)).fetchall()


def approximate_vector_index(connection) -> None:
    connection.execute("CREATE INDEX v_idx ON v (libsql_vector_idx(vector))")
    connection.execute("SELECT id FROM vector_top_k('v_idx', vector32('[1,0]'), 1)").fetchall()


def quoted_identifier_case_in_sqlite_master(connection) -> None:
    connection.execute('CREATE TABLE "Doc_text" (id TEXT PRIMARY KEY)')
    names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master").fetchall()]
    if "Doc_text" not in names:
        raise AssertionError(f"stored lowercased: {names}")


def parenthesized_join_in_from(connection) -> None:
    connection.execute(
        "SELECT * FROM e JOIN (v JOIN e AS e2 ON e2.source_id = v.id) ON e.target_id = v.id"
    ).fetchall()


def sqlalchemy_aioturso_dialect() -> None:
    import asyncio

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    async def probe():
        path = os.path.join(tempfile.mkdtemp(), "d.db")
        engine = create_async_engine(f"sqlite+aioturso:///{path}")
        async with engine.begin() as connection:
            await connection.execute(text("SELECT 1"))
        await engine.dispose()

    asyncio.run(probe())


def upstream_reflection_stubs() -> None:
    from turso.sqlalchemy.dialect import _TursoDialectMixin

    if "get_indexes" in _TursoDialectMixin.__dict__:
        raise AssertionError("dialect mixin stubs get_indexes/get_foreign_keys to []")


REPRODUCTIONS: list[tuple[str, Callable]] = [
    ("WITH RECURSIVE", lambda: recursive_cte(_fresh())),
    (
        "scalar subquery in ON CONFLICT DO UPDATE SET",
        lambda: scalar_subquery_in_upsert_set(_fresh()),
    ),
    ("bind parameter inside nested json_each(?)", lambda: bind_inside_nested_json_each(_fresh())),
    ("datetime bind parameter", lambda: datetime_bind_parameter(_fresh())),
    ("libsql_vector_idx / vector_top_k", lambda: approximate_vector_index(_fresh())),
    (
        "quoted identifier case in sqlite_master",
        lambda: quoted_identifier_case_in_sqlite_master(_fresh()),
    ),
    ("parenthesized join in FROM clause", lambda: parenthesized_join_in_from(_fresh())),
    ("sqlite+aioturso dialect on this SQLAlchemy", sqlalchemy_aioturso_dialect),
    ("upstream dialect reflection stubs", upstream_reflection_stubs),
]


# What a reproduction may raise: the driver's DB-API errors, SQLAlchemy's
# wrappers, the AttributeError of the upstream dialect bug, or this script's
# own assertions about engine behaviour.
def _reproduction_errors() -> tuple[type[BaseException], ...]:
    import turso
    from sqlalchemy.exc import SQLAlchemyError

    return (turso.Error, SQLAlchemyError, AttributeError, AssertionError)


def main() -> None:
    import turso

    expected_errors = _reproduction_errors()

    print(
        f"pyturso {importlib.metadata.version('pyturso')} | turso_version() = "
        f"{turso.connect(':memory:').execute('SELECT turso_version()').fetchone()[0]}"
    )
    for name, reproduction in REPRODUCTIONS:
        try:
            reproduction()
        except expected_errors as error:
            print(f"GAP  {name}: {type(error).__name__}: {str(error).splitlines()[0][:110]}")
        else:
            print(f"OK   {name}")


if __name__ == "__main__":
    main()
