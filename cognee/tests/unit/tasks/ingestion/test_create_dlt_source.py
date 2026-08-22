import pytest

from cognee.tasks.ingestion.create_dlt_source import _parse_sql_query


@pytest.mark.parametrize(
    "query,expected",
    [
        # Bare table names keep working exactly as before.
        ("SELECT * FROM users", ("users", "1=1")),
        ("SELECT id, name FROM users WHERE age > 18", ("users", "age > 18")),
        # Schema-qualified names must be captured in full, not truncated at the
        # dot (previously returned just the schema and dropped the WHERE clause).
        ("SELECT * FROM public.users WHERE age > 18", ("public.users", "age > 18")),
        ("SELECT id FROM analytics.events", ("analytics.events", "1=1")),
        (
            "SELECT a, b FROM my_db.schema.table WHERE x = 1 AND y = 2",
            ("my_db.schema.table", "x = 1 AND y = 2"),
        ),
        # The non-greedy FROM split must not trip on FROM inside the WHERE clause.
        (
            "SELECT * FROM users WHERE note = 'imported FROM legacy'",
            ("users", "note = 'imported FROM legacy'"),
        ),
        (
            "SELECT * FROM orders WHERE customer_id IN (SELECT id FROM vip_customers)",
            ("orders", "customer_id IN (SELECT id FROM vip_customers)"),
        ),
        # Keywords are case-insensitive.
        ("select id from public.users where active = true", ("public.users", "active = true")),
    ],
)
def test_parse_sql_query(query, expected):
    assert _parse_sql_query(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM users",
        "UPDATE users SET active = false",
        "SELECT 1 + 1",
    ],
)
def test_parse_sql_query_rejects_invalid_input(query):
    with pytest.raises(ValueError):
        _parse_sql_query(query)
