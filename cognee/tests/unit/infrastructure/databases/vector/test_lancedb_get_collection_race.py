"""get_collection survives a table that is listed but not yet committed.

Two data items of one cognify run index the same collection at the same
moment: one is inside create_collection (under VECTOR_DB_LOCK) when the other
lists the table directory, sees it, and opens it before the first manifest is
written. LanceDB answers "Table 'X' was not found" to that open. The adapter
retries the open without taking the lock, so a caller that already holds it
(prune, create_data_points) cannot deadlock on itself.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.lancedb import LanceDBAdapter as adapter_module
from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(adapter_module, "_OPEN_TABLE_RETRY_DELAYS", (0.01, 0.01, 0.01))


def _adapter(connection) -> LanceDBAdapter:
    adapter = object.__new__(LanceDBAdapter)
    adapter.VECTOR_DB_LOCK = asyncio.Lock()
    adapter.get_connection = AsyncMock(return_value=connection)
    return adapter


@pytest.mark.asyncio
async def test_a_half_created_table_is_opened_once_its_manifest_is_committed():
    table = SimpleNamespace(name="EntityType_name")
    connection = SimpleNamespace(
        table_names=AsyncMock(return_value=["EntityType_name"]),
        open_table=AsyncMock(
            side_effect=[ValueError("Table 'EntityType_name' was not found"), table]
        ),
    )
    adapter = _adapter(connection)

    assert await adapter.get_collection("EntityType_name") is table
    assert connection.open_table.await_count == 2


@pytest.mark.asyncio
async def test_a_caller_holding_the_lock_does_not_deadlock():
    """prune() holds VECTOR_DB_LOCK while it opens every table, a stray one included."""
    connection = SimpleNamespace(
        table_names=AsyncMock(return_value=["Stray"]),
        open_table=AsyncMock(side_effect=ValueError("Table 'Stray' was not found")),
    )
    adapter = _adapter(connection)

    async with adapter.VECTOR_DB_LOCK:
        with pytest.raises(CollectionNotFoundError):
            await asyncio.wait_for(adapter.get_collection("Stray"), timeout=5)
    assert connection.open_table.await_count == 3


@pytest.mark.asyncio
async def test_a_table_that_never_gets_a_manifest_is_reported_missing():
    """A creator that crashed mid-create leaves a directory with no manifest; callers
    catch CollectionNotFoundError, not lance's own error."""
    connection = SimpleNamespace(
        table_names=AsyncMock(return_value=["Orphan"]),
        open_table=AsyncMock(side_effect=ValueError("Table 'Orphan' was not found")),
    )
    adapter = _adapter(connection)

    assert await adapter.retrieve("Orphan", ["id-1"]) == []


@pytest.mark.asyncio
async def test_a_table_that_is_really_missing_still_raises_collection_not_found():
    connection = SimpleNamespace(
        table_names=AsyncMock(side_effect=[["EntityType_name"], []]),
        open_table=AsyncMock(side_effect=ValueError("Table 'EntityType_name' was not found")),
    )
    adapter = _adapter(connection)

    with pytest.raises(CollectionNotFoundError):
        await adapter.get_collection("EntityType_name")


@pytest.mark.asyncio
async def test_other_value_errors_are_not_retried():
    connection = SimpleNamespace(
        table_names=AsyncMock(return_value=["EntityType_name"]),
        open_table=AsyncMock(side_effect=ValueError("schema mismatch")),
    )
    adapter = _adapter(connection)

    with pytest.raises(ValueError, match="schema mismatch"):
        await adapter.get_collection("EntityType_name")
    assert connection.open_table.await_count == 1
