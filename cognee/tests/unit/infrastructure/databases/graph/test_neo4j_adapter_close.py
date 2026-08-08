import sys
import gc
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Ensure neo4j and related modules are mocked if they are not installed in the environment
try:
    import neo4j
except ModuleNotFoundError:
    sys.modules["neo4j"] = MagicMock()
    sys.modules["neo4j.exceptions"] = MagicMock()

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
from cognee.infrastructure.databases.utils.closing_lru_cache import ClosingLRUCache

@pytest.mark.asyncio
async def test_neo4j_adapter_close():
    # Arrange: Create a mock driver with an async close method
    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()

    # Instantiate the Neo4jAdapter using the mock driver
    adapter = Neo4jAdapter(
        graph_database_url="bolt://localhost:7687",
        graph_database_username="neo4j",
        graph_database_password="password",
        driver=mock_driver,
    )

    # Act: Call close()
    await adapter.close()

    # Assert: Verify that self.driver.close was called and awaited
    mock_driver.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_neo4j_adapter_close_none_driver():
    # Arrange: Instantiate with no driver (it would normally call AsyncGraphDatabase.driver)
    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()

    with patch("cognee.infrastructure.databases.graph.neo4j_driver.adapter.AsyncGraphDatabase") as mock_db:
        mock_db.driver.return_value = mock_driver
        adapter = Neo4jAdapter(
            graph_database_url="bolt://localhost:7687",
            graph_database_username="neo4j",
            graph_database_password="password",
        )

        # Act: Call close()
        await adapter.close()

        # Assert: Verify that driver.close was called and awaited
        mock_driver.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_closing_lru_cache_evicts_neo4j_adapter():
    # Arrange: Create cache with maxsize=1 so adding a second item triggers eviction
    cache = ClosingLRUCache(maxsize=1)
    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()

    adapter = Neo4jAdapter(
        graph_database_url="bolt://localhost:7687",
        graph_database_username="neo4j",
        graph_database_password="password",
        driver=mock_driver,
    )

    # Put the adapter in cache
    proxy = cache.get_or_create("neo4j_key", lambda: adapter)
    
    # Verify the proxy wraps our adapter
    assert proxy.__wrapped__ is adapter
    
    # Act: evict by putting a new item in the cache
    cache.get_or_create("another_key", lambda: MagicMock())
    
    # At this point, the proxy is still held in 'proxy' variable, so it shouldn't close yet
    assert mock_driver.close.call_count == 0

    # Release the lease
    del proxy
    gc.collect()

    # Allow any pending async close tasks to complete
    await asyncio.sleep(0.1)

    # Assert: driver should be closed
    mock_driver.close.assert_awaited_once()
