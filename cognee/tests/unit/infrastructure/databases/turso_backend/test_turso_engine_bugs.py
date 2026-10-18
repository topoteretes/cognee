"""Reproductions of known pyturso engine bugs, checked against stock SQLite.

Each test runs the same statements on the stdlib ``sqlite3`` and on pyturso and
expects the same rows. They are strict xfails: a pyturso release that fixes a bug
makes its test pass, which fails the run and flags the xfail for removal.
"""

import sqlite3

import pytest

turso = pytest.importorskip("turso", reason="pyturso not installed")


@pytest.mark.xfail(
    strict=True,
    reason="pyturso drops the u.id = a.id join condition when inner joins, "
    "a constant filter and a LEFT JOIN are combined",
)
def test_inner_joins_before_a_left_join_keep_every_join_condition():
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
