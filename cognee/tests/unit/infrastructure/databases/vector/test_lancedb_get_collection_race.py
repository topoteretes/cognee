"""get_collection survives a table that is listed but not yet committed.

Two data items of one cognify run index the same collection at the same
moment: one is inside create_collection (under VECTOR_DB_LOCK) when the other
lists the table directory, sees it, and opens it before the first manifest is
written. LanceDB answers "Table 'X' was not found" to that open. The adapter
waits for the creator by taking the same lock and opens once more.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter


def _adapter(connection) -> LanceDBAdapter:
    adapter = object.__new__(LanceDBAdapter)
    adapter.VECTOR_DB_LOCK = asyncio.Lock()
    adapter.get_connection = AsyncMock(return_value=connection)
    return adapter


@pytest.mark.asyncio
async def test_a_half_created_table_is_opened_once_the_creator_is_done():
    table = SimpleNamespace(name="EntityType_name")
    connection = SimpleNamespace(
        table_names=AsyncMock(return_value=["EntityType_name"]),
        open_table=AsyncMock(
            side_effect=[ValueError("Table 'EntityType_name' was not found"), table]
        ),
    )
    adapter = _adapter(connection)

    # The creator still holds the lock when the first open fails; it is
    # released shortly after, as create_collection does when create_table returns.
    await adapter.VECTOR_DB_LOCK.acquire()
    asyncio.get_running_loop().call_later(0.05, adapter.VECTOR_DB_LOCK.release)

    assert await adapter.get_collection("EntityType_name") is table
    assert connection.open_table.await_count == 2


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
