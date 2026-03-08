import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.infrastructure.databases.graph.spanner.adapter import SpannerAdapter

@pytest.fixture
def mock_spanner_client():
    with patch("cognee.infrastructure.databases.graph.spanner.adapter.spanner.Client") as mock_client:
        mock_instance = MagicMock()
        mock_database = MagicMock()
        mock_client.return_value.instance.return_value = mock_instance
        mock_instance.database.return_value = mock_database
        yield mock_client

@pytest.mark.asyncio
async def test_spanner_adapter_init(mock_spanner_client):
    adapter = SpannerAdapter("test-project", "test-instance", "test-database")
    mock_spanner_client.assert_called_once_with(project="test-project")
    mock_spanner_client.return_value.instance.assert_called_once_with("test-instance")
    mock_spanner_client.return_value.instance.return_value.database.assert_called_once_with("test-database")

@pytest.mark.asyncio
async def test_spanner_add_node(mock_spanner_client):
    adapter = SpannerAdapter("test-project", "test-instance", "test-database")
    mock_run_in_transaction = mock_spanner_client.return_value.instance.return_value.database.run_in_transaction
    
    await adapter.add_node("node1", {"name": "Test Node", "type": "TEST"})
    mock_run_in_transaction.assert_called_once()
