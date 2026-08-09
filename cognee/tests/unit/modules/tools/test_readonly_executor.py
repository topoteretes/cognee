import sqlite3
from datetime import datetime

import pytest
from sqlalchemy import text

from cognee.modules.tools.text_to_sql.engine_factory import (
    get_external_sql_engine,
    introspect_schema,
)
from cognee.modules.tools.text_to_sql.executor import execute_readonly


@pytest.fixture
def sample_db(tmp_path):
    db_path = tmp_path / "external.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            region TEXT NOT NULL,
            amount REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO orders (region, amount, created_at) VALUES
            ('eu', 10.5, '2026-01-01T00:00:00'),
            ('eu', 20.0, '2026-01-02T00:00:00'),
            ('us', 30.0, '2026-01-03T00:00:00');
        """
    )
    connection.commit()
    connection.close()
    return db_path


def _engine(db_path):
    return get_external_sql_engine(f"sqlite:///{db_path}", "sqlite", 5000)


@pytest.mark.asyncio
async def test_select_returns_json_safe_rows(sample_db):
    rows, truncated = await execute_readonly(
        _engine(sample_db), "SELECT region, amount FROM orders ORDER BY id", max_rows=10
    )
    assert truncated is False
    assert rows[0] == {"region": "eu", "amount": 10.5}
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_truncation_flag_and_row_cap(sample_db):
    rows, truncated = await execute_readonly(_engine(sample_db), "SELECT * FROM orders", max_rows=2)
    assert truncated is True
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_empty_result_is_not_an_error(sample_db):
    rows, truncated = await execute_readonly(
        _engine(sample_db), "SELECT * FROM orders WHERE region = 'jp'", max_rows=10
    )
    assert rows == []
    assert truncated is False


@pytest.mark.asyncio
async def test_write_blocked_at_database_level(sample_db):
    """The sqlite engine is opened with mode=ro — writes fail in the DB itself."""
    engine = _engine(sample_db)
    with pytest.raises(Exception) as exc_info:
        async with engine.connect() as connection:
            await connection.execute(
                text("INSERT INTO orders (region, amount, created_at) VALUES ('x', 1, 'now')")
            )
    assert "readonly" in str(exc_info.value).lower() or "read-only" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_json_coercion_of_python_types(sample_db):
    from cognee.modules.tools.text_to_sql.executor import _json_safe
    from decimal import Decimal
    from uuid import uuid4

    assert _json_safe(Decimal("1.5")) == 1.5
    assert _json_safe(datetime(2026, 1, 1)) == "2026-01-01T00:00:00"
    uid = uuid4()
    assert _json_safe(uid) == str(uid)
    assert _json_safe(b"\x00\x01") == "AAE="
    assert _json_safe(None) is None
    assert _json_safe("plain") == "plain"


@pytest.mark.asyncio
async def test_introspect_schema(sample_db):
    schema = await introspect_schema(_engine(sample_db))
    assert "orders" in schema
    column_names = {column["name"] for column in schema["orders"]["columns"]}
    assert {"id", "region", "amount", "created_at"} <= column_names
    assert schema["orders"]["primary_key"] == ["id"]


@pytest.mark.asyncio
async def test_introspect_schema_respects_allowed_tables_and_cap(sample_db):
    schema = await introspect_schema(_engine(sample_db), allowed_tables=["nope"])
    assert schema == {}
    schema = await introspect_schema(_engine(sample_db), max_tables=0)
    assert schema == {}
