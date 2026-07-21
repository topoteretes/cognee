"""Unit tests for ``cognee.tasks.ingestion.create_dlt_source._parse_sql_query``.

The parser feeds ``dlt``'s ``sql_database`` adapter: the captured ``table_name``
selects which table to load and the ``where_clause`` filters rows. A silent
mispin (wrong table or dropped WHERE) means the user gets back the whole table
instead of the slice they asked for. These tests pin the parser against:

  - bare table names (the original happy path)
  - schema-qualified table names (regression for #3663)
  - WHERE clauses with subqueries / quoted literals
  - non-SELECT statements (must raise so the caller surfaces the bad input)
"""

import pytest

from cognee.tasks.ingestion.create_dlt_source import _parse_sql_query


def test_bare_table_name_no_where():
    assert _parse_sql_query("SELECT * FROM users") == ("users", "1=1")


def test_bare_table_name_with_where():
    assert _parse_sql_query("SELECT * FROM users WHERE age > 18") == (
        "users",
        "age > 18",
    )


def test_schema_qualified_table_name_no_where():
    # Regression for #3663: previously captured "public" and dropped ".users".
    assert _parse_sql_query("SELECT * FROM public.users") == ("public.users", "1=1")


def test_schema_qualified_table_name_with_where():
    # Regression for #3663: the WHERE clause was silently dropped when the
    # table name contained a dot, because the (\w+) capture ended at the dot
    # and the trailing ".users WHERE ..." no longer matched the WHERE group.
    assert _parse_sql_query("SELECT * FROM public.users WHERE age > 18") == (
        "public.users",
        "age > 18",
    )


def test_catalog_schema_table_name():
    assert _parse_sql_query("SELECT id FROM analytics.events") == (
        "analytics.events",
        "1=1",
    )


def test_where_clause_with_quoted_string_containing_from():
    # The non-greedy "FROM" splitter must not trip on "FROM" appearing inside
    # a quoted literal in the WHERE clause.
    assert _parse_sql_query(
        "SELECT * FROM users WHERE note = 'imported FROM legacy'"
    ) == ("users", "note = 'imported FROM legacy'")


def test_where_clause_with_subquery():
    assert _parse_sql_query(
        "SELECT * FROM orders WHERE customer_id IN (SELECT id FROM vip_customers)"
    ) == (
        "orders",
        "customer_id IN (SELECT id FROM vip_customers)",
    )


def test_case_insensitive_keywords():
    assert _parse_sql_query("select id from public.users where active = true") == (
        "public.users",
        "active = true",
    )


def test_leading_trailing_whitespace_is_tolerated():
    assert _parse_sql_query("  SELECT * FROM users\n") == ("users", "1=1")


def test_non_select_query_raises():
    with pytest.raises(ValueError):
        _parse_sql_query("UPDATE users SET active = false")


def test_missing_from_raises():
    with pytest.raises(ValueError):
        _parse_sql_query("SELECT 1 + 1")
