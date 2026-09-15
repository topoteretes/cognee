"""Unit tests for DLT SQL query parsing.

``_parse_sql_query`` is a pure function used by
``create_dlt_source_from_connection_string`` to pick a table and WHERE
filter. Schema-qualified names (``public.users``) used to be truncated at
the first dot, which also dropped the WHERE clause.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cognee.tasks.ingestion.create_dlt_source import (
    _parse_sql_query,
    _split_schema_and_table,
    create_dlt_source_from_connection_string,
)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("SELECT * FROM users", ("users", "1=1")),
        ("select id from users", ("users", "1=1")),
        (
            "SELECT * FROM public.users WHERE age > 18",
            ("public.users", "age > 18"),
        ),
        ("SELECT id FROM analytics.events", ("analytics.events", "1=1")),
        (
            "SELECT id FROM analytics.events WHERE ts > 0",
            ("analytics.events", "ts > 0"),
        ),
        (
            "  SELECT * FROM users WHERE name = 'from town'  ",
            ("users", "name = 'from town'"),
        ),
        (
            "SELECT a, b FROM reporting.fact_orders WHERE status = 'open'",
            ("reporting.fact_orders", "status = 'open'"),
        ),
        # A table alias must not break WHERE adjacency (#4839): the alias
        # segment used to defeat the WHERE group, silently falling back to 1=1
        # and ingesting the whole table.
        (
            "SELECT * FROM users AS u WHERE age > 18",
            ("users", "age > 18"),
        ),
        ("SELECT * FROM users u WHERE age > 18", ("users", "age > 18")),
        ("SELECT * FROM users u", ("users", "1=1")),
        (
            "SELECT * FROM public.users AS u WHERE age > 18",
            ("public.users", "age > 18"),
        ),
        # An alias named like a clause keyword is a clause keyword, not an alias.
        (
            "SELECT * FROM users ORDER BY age LIMIT 5",
            ("users", "1=1"),
        ),
    ],
)
def test_parse_sql_query(query, expected):
    assert _parse_sql_query(query) == expected


def test_parse_sql_query_rejects_non_select():
    with pytest.raises(ValueError, match="Cannot parse SQL query"):
        _parse_sql_query("DELETE FROM users")


def test_parse_sql_query_rejects_alias_qualified_where():
    """dlt replays the filter against the bare table, so an alias reference cannot work."""
    with pytest.raises(ValueError, match="references the table alias 'u'"):
        _parse_sql_query("SELECT * FROM users u WHERE u.age > 18")


def test_parse_sql_query_rejects_joins():
    """A JOIN query's WHERE spans tables the single-table source cannot express."""
    with pytest.raises(ValueError, match="JOIN queries are not supported"):
        _parse_sql_query(
            "SELECT o.id FROM orders o JOIN customers c ON o.cid = c.id WHERE c.age > 18"
        )
    with pytest.raises(ValueError, match="JOIN queries are not supported"):
        _parse_sql_query("SELECT * FROM users LEFT JOIN roles r ON users.rid = r.id")


@pytest.mark.parametrize(
    "qualified, expected",
    [
        ("users", (None, "users")),
        ("public.users", ("public", "users")),
        ("analytics.events", ("analytics", "events")),
        ("catalog.schema.table", ("catalog.schema", "table")),
    ],
)
def test_split_schema_and_table(qualified, expected):
    assert _split_schema_and_table(qualified) == expected


def _install_fake_sql_database(monkeypatch):
    """Stub dlt so the connection-string helper can be tested without the extra."""
    mock_sql = MagicMock(return_value=MagicMock(name="source"))
    fake_sql_database = ModuleType("dlt.sources.sql_database")
    fake_sql_database.sql_database = mock_sql
    monkeypatch.setitem(sys.modules, "dlt", ModuleType("dlt"))
    monkeypatch.setitem(sys.modules, "dlt.sources", ModuleType("dlt.sources"))
    monkeypatch.setitem(sys.modules, "dlt.sources.sql_database", fake_sql_database)
    return mock_sql


def test_schema_qualified_query_passes_schema_and_bare_table_to_dlt(monkeypatch):
    mock_sql = _install_fake_sql_database(monkeypatch)

    create_dlt_source_from_connection_string(
        "postgresql://user:pass@localhost/db",
        query="SELECT * FROM public.users WHERE age > 18",
    )

    kwargs = mock_sql.call_args.kwargs
    assert kwargs["table_names"] == ["users"]
    assert kwargs["schema"] == "public"

    query = MagicMock()
    matching = SimpleNamespace(name="users")
    other = SimpleNamespace(name="orders")
    kwargs["query_adapter_callback"](query, matching)
    query.where.assert_called_once()
    query.reset_mock()
    assert kwargs["query_adapter_callback"](query, other) is query
    query.where.assert_not_called()


def test_bare_table_query_does_not_set_schema(monkeypatch):
    mock_sql = _install_fake_sql_database(monkeypatch)

    create_dlt_source_from_connection_string(
        "postgresql://user:pass@localhost/db",
        query="SELECT * FROM users WHERE age > 18",
    )

    kwargs = mock_sql.call_args.kwargs
    assert kwargs["table_names"] == ["users"]
    assert "schema" not in kwargs
