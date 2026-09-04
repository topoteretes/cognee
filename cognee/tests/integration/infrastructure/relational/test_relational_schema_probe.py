"""Integration tests for ``_relational_schema_exists`` against REAL SQLite stores.

The probe decides which bootstrap branch a database takes: *exists* ->
``alembic upgrade head`` (migration bodies run), *empty* -> ``create_all`` +
``stamp head`` (no bodies run). These tests lock in its tri-state contract:

- a missing local SQLite file is FRESH, decided from the filesystem without
  ever connecting (the probe must not create the file as a side effect);
- a reachable database with no cognee tables is FRESH via inspection;
- a database holding ``users`` or ``alembic_version`` EXISTS;
- an uninspectable database RAISES — "cannot inspect" must never be answered
  as "empty", because that answer sends a populated database into the
  create_all + stamp-head bootstrap and mis-stamps it at head with no
  migration having run.

Only the relational engine is touched: the adapter is built directly on a tmp
path and injected by patching ``get_relational_engine`` — no config, LLM, or
graph/vector stores involved.
"""

import asyncio
import os
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.migrations.startup import _relational_schema_exists

_GET_ENGINE = "cognee.infrastructure.databases.relational.get_relational_engine"


def _probe_with(adapter: SQLAlchemyAdapter) -> bool:
    async def _run():
        with patch(_GET_ENGINE, return_value=adapter):
            try:
                return await _relational_schema_exists()
            finally:
                await adapter.engine.dispose()

    return asyncio.run(_run())


def _adapter_for(db_path) -> SQLAlchemyAdapter:
    return SQLAlchemyAdapter(f"sqlite+aiosqlite:///{db_path}")


def test_missing_sqlite_file_is_fresh_and_probe_does_not_create_it(tmp_path):
    db_path = tmp_path / "not_created_yet.db"

    assert _probe_with(_adapter_for(db_path)) is False
    # Decided from the filesystem, not by connecting: an aiosqlite connect
    # would have created the file.
    assert not os.path.exists(db_path)


def test_reachable_empty_database_is_fresh_via_inspection(tmp_path):
    db_path = tmp_path / "empty.db"
    db_path.touch()  # existing zero-byte file is a valid empty SQLite database

    assert _probe_with(_adapter_for(db_path)) is False
    assert os.path.exists(db_path)


@pytest.mark.parametrize("marker_table", ["users", "alembic_version"])
def test_database_with_either_marker_table_exists(tmp_path, marker_table):
    db_path = tmp_path / "populated.db"
    seed = _adapter_for(db_path)

    async def _seed():
        async with seed.engine.begin() as connection:
            await connection.execute(text(f"CREATE TABLE {marker_table} (id INTEGER PRIMARY KEY)"))
        await seed.engine.dispose()

    asyncio.run(_seed())

    assert _probe_with(_adapter_for(db_path)) is True


def test_uninspectable_database_raises_instead_of_reporting_fresh(tmp_path):
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"this is not a sqlite database, inspection must fail")

    adapter = _adapter_for(db_path)
    with pytest.raises(DatabaseError):
        _probe_with(adapter)
