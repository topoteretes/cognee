"""Reproductions of pyturso engine bugs, checked against stock SQLite.

Each test runs the same statements on the stdlib ``sqlite3`` and on pyturso and
expects the same rows. A bug the pinned release still has is a strict xfail that
accepts only the known failure: any other error fails the test, and a pyturso
release that fixes the bug makes the test pass, which fails the run and flags the
xfail for removal. A bug a release fixed stays as a plain regression test.
"""

import sqlite3
import subprocess
import sys

import pytest

turso = pytest.importorskip("turso", reason="pyturso not installed")


def test_inner_joins_before_a_left_join_keep_every_join_condition():
    """pyturso 0.7.2 dropped ``u.id = a.id`` here and returned ``(2, 1)``; fixed in 0.8."""
    expected = _join_chain_rows(sqlite3.connect(":memory:"))
    assert expected == []
    assert _join_chain_rows(turso.connect(":memory:")) == expected


def _join_chain_rows(connection) -> list[tuple]:
    """Run the smallest known failing join chain on one-row tables (``b`` is empty)."""
    for table, row in (("u", (2, 1)), ("a", (1, 1)), ("p", (3, 1)), ("b", None)):
        connection.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER)")
        if row is not None:
            connection.execute(f"INSERT INTO {table} VALUES (?, ?)", row)
    rows = connection.execute(
        "SELECT u.id, a.id FROM u JOIN a ON u.id = a.id AND a.k = 1 "
        "JOIN p ON p.k = a.k LEFT JOIN b ON u.k = b.k"
    ).fetchall()
    connection.close()
    return sorted(tuple(row) for row in rows)


_RECURSIVE_CTE_READ_TWICE = """
import {module} as engine
connection = engine.connect(":memory:")
connection.execute("CREATE TABLE e (s TEXT, t TEXT)")
connection.execute("INSERT INTO e VALUES ('a', 'b'), ('b', 'c')")
rows = connection.execute(
    "WITH RECURSIVE r(id) AS (SELECT value FROM json_each('[\\"a\\"]') "
    "UNION SELECT e.t FROM r JOIN e ON e.s = r.id) "
    "SELECT s, t FROM e WHERE s IN (SELECT id FROM r) AND t IN (SELECT id FROM r)"
).fetchall()
print(sorted(tuple(row) for row in rows))
"""


class KnownEnginePanic(Exception):
    """The child process died with the engine panic the xfail below is about."""


@pytest.mark.xfail(
    strict=True,
    raises=KnownEnginePanic,
    reason="pyturso 0.8.3 aborts the process ('duplicate cursor key') on a recursive CTE "
    "seeded from json_each() that the query reads twice, the shape of a k-hop "
    "neighborhood query; the graph adapter walks one query per hop instead",
)
def test_recursive_cte_seeded_from_json_each_can_be_read_twice():
    """Runs in a subprocess: the engine panics, which would take the test run with it.

    Only that panic is the expected failure. Any other child error or wrong rows
    fails the test.
    """
    expected = _run_recursive_cte("sqlite3")
    assert expected.returncode == 0 and expected.stdout.strip() == "[('a', 'b'), ('b', 'c')]"
    actual = _run_recursive_cte("turso")
    if actual.returncode < 0 and "duplicate cursor key" in actual.stderr:
        raise KnownEnginePanic(actual.stderr)
    assert actual.returncode == 0, actual.stderr
    assert actual.stdout == expected.stdout


def _run_recursive_cte(module: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _RECURSIVE_CTE_READ_TWICE.format(module=module)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
