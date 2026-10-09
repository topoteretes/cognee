import pytest

from cognee.modules.tools.errors import SqlGuardError
from cognee.modules.tools.text_to_sql.write_guard import validate_update


def test_plain_update_with_where_passes():
    sql, table = validate_update("UPDATE orders SET region = 'eu' WHERE id = 42")
    assert table == "orders"
    assert sql.startswith("UPDATE")


def test_markdown_and_semicolon_cleaned():
    sql, table = validate_update("```sql\nUPDATE orders SET amount = 1 WHERE id = 1;\n```")
    assert table == "orders"
    assert not sql.endswith(";")


def test_missing_where_rejected():
    with pytest.raises(SqlGuardError, match="WHERE"):
        validate_update("UPDATE orders SET region = 'eu'")


def test_non_update_statements_rejected():
    for bad in (
        "SELECT * FROM orders",
        "DELETE FROM orders WHERE id = 1",
        "DROP TABLE orders",
        "INSERT INTO orders VALUES (1)",
        "",
    ):
        with pytest.raises(SqlGuardError):
            validate_update(bad)


def test_multi_statement_rejected():
    with pytest.raises(SqlGuardError):
        validate_update("UPDATE orders SET a = 1 WHERE id = 1; DROP TABLE orders")


def test_forbidden_keyword_rejected():
    with pytest.raises(SqlGuardError, match="RETURNING"):
        validate_update("UPDATE orders SET a = 1 WHERE id = 1 RETURNING *")


def test_forbidden_word_inside_string_literal_passes():
    sql, _ = validate_update("UPDATE logs SET note = 'please delete this' WHERE id = 1")
    assert "delete" in sql


def test_where_only_in_string_literal_rejected():
    with pytest.raises(SqlGuardError, match="WHERE"):
        validate_update("UPDATE logs SET note = 'where am i'")


def test_allowed_tables_enforced():
    validate_update("UPDATE orders SET a = 1 WHERE id = 1", allowed_tables=["orders"])
    with pytest.raises(SqlGuardError, match="allowed"):
        validate_update("UPDATE users SET a = 1 WHERE id = 1", allowed_tables=["orders"])


def test_quoted_and_qualified_table_names():
    _, table = validate_update('UPDATE "orders" SET a = 1 WHERE id = 1')
    assert table == "orders"
    _, table = validate_update("UPDATE public.orders SET a = 1 WHERE id = 1")
    assert table == "public.orders"
