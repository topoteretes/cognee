import sys
import gc
import asyncio
from unittest.mock import MagicMock, patch
import pytest

# Mock langchain_aws if not installed
if "langchain_aws" not in sys.modules:
    mock_lc = MagicMock()
    sys.modules["langchain_aws"] = mock_lc

from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB
from cognee.infrastructure.databases.utils.closing_lru_cache import ClosingLRUCache


@pytest.mark.asyncio
async def test_neptune_adapter_close():
    """close() should clean up the underlying boto3 client and set _client to None."""
    # Arrange: create a mock NeptuneAnalyticsGraph with a nested boto3 client
    mock_boto3_client = MagicMock()
    mock_boto3_client.close = MagicMock()

    mock_neptune_graph = MagicMock()
    mock_neptune_graph.client = mock_boto3_client

    with patch(
        "cognee.infrastructure.databases.graph.neptune_driver.adapter.NeptuneGraphDB._initialize_client",
        return_value=mock_neptune_graph,
    ):
        adapter = NeptuneGraphDB(
            graph_id="g-1234",
            region="us-east-1",
        )

    # Act
    await adapter.close()

    # Assert: boto3 client closed, _client set to None
    mock_boto3_client.close.assert_called_once()
    assert adapter._client is None


@pytest.mark.asyncio
async def test_neptune_adapter_close_no_client():
    """close() should be a no-op when _client is None."""
    with patch(
        "cognee.infrastructure.databases.graph.neptune_driver.adapter.NeptuneGraphDB._initialize_client",
        return_value=None,
    ), patch(
        "cognee.infrastructure.databases.graph.neptune_driver.adapter.NeptuneGraphDB.__init__",
        side_effect=lambda **kw: None,
    ):
        adapter = NeptuneGraphDB.__new__(NeptuneGraphDB)
        adapter._client = None

    # Should not raise
    await adapter.close()
    assert adapter._client is None


@pytest.mark.asyncio
async def test_neptune_adapter_close_no_underlying_client():
    """close() should handle _client without a .client attribute gracefully."""
    mock_neptune_graph = MagicMock(spec=[])  # no .client attribute

    adapter = NeptuneGraphDB.__new__(NeptuneGraphDB)
    adapter._client = mock_neptune_graph

    # Should not raise
    await adapter.close()
    assert adapter._client is None


@pytest.mark.asyncio
async def test_closing_lru_cache_evicts_neptune_adapter():
    """When evicted from closing_lru_cache, the adapter's close() should be called."""
    cache = ClosingLRUCache(maxsize=1)

    mock_boto3_client = MagicMock()
    mock_boto3_client.close = MagicMock()

    mock_neptune_graph = MagicMock()
    mock_neptune_graph.client = mock_boto3_client

    with patch(
        "cognee.infrastructure.databases.graph.neptune_driver.adapter.NeptuneGraphDB._initialize_client",
        return_value=mock_neptune_graph,
    ):
        adapter = NeptuneGraphDB(
            graph_id="g-1234",
            region="us-east-1",
        )

    proxy = cache.get_or_create("neptune_key", lambda: adapter)
    assert proxy.__wrapped__ is adapter

    # Evict by adding another entry
    cache.get_or_create("another_key", lambda: MagicMock())

    # Proxy still held, so not closed yet
    assert mock_boto3_client.close.call_count == 0

    # Release the lease
    del proxy
    gc.collect()

    # Allow pending async close tasks to complete
    await asyncio.sleep(0.1)

    # Now the boto3 client should be closed
    mock_boto3_client.close.assert_called_once()
