"""``create_db_and_tables`` runs on every add() and every pipeline run, so it must
not rebuild (alembic upgrade head) an existing schema each time — that made every
request log a full migration block. Only an empty database is created."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import cognee.modules.migrations.startup as startup

# The package re-exports the function under the module's name; import the module.
module = importlib.import_module("cognee.infrastructure.databases.relational.create_db_and_tables")


@pytest.mark.asyncio
async def test_existing_schema_is_left_alone():
    engine = MagicMock(create_database=AsyncMock())
    with (
        patch.object(startup, "_relational_schema_exists", new=AsyncMock(return_value=True)),
        patch.object(module, "get_relational_engine", return_value=engine),
    ):
        await module.create_db_and_tables()
    engine.create_database.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_database_is_created():
    engine = MagicMock(create_database=AsyncMock())
    with (
        patch.object(startup, "_relational_schema_exists", new=AsyncMock(return_value=False)),
        patch.object(module, "get_relational_engine", return_value=engine),
    ):
        await module.create_db_and_tables()
    engine.create_database.assert_awaited_once()
