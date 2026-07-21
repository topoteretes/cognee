import pytest
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from cognee.tasks.memify.decay_memory import decay_memory

class MockData:
    def __init__(self, id, last_accessed=None):
        self.id = id
        self.last_accessed = last_accessed

class MockDatasetData:
    def __init__(self, data_id, dataset_id):
        self.data_id = data_id
        self.dataset_id = dataset_id

class MockNode:
    def __init__(self, slug):
        self.slug = slug

class MockEdge:
    def __init__(self, slug):
        self.slug = slug

@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"ENABLE_LAST_ACCESSED": "true"}):
        yield

@pytest.mark.asyncio
async def test_decay_memory_skipped_if_disabled():
    with patch.dict(os.environ, {"ENABLE_LAST_ACCESSED": "false"}):
        result = await decay_memory()
        assert result["status"] == "skipped"
        assert result["reason"] == "ENABLE_LAST_ACCESSED not enabled"

@pytest.mark.asyncio
async def test_decay_memory_dry_run(mock_env):
    mock_db = MagicMock()
    mock_session = AsyncMock()
    mock_db.get_async_session.return_value.__aenter__.return_value = mock_session
    
    # Mock query result
    mock_result = MagicMock()
    data_id = uuid4()
    dataset_id = uuid4()
    mock_result.all.return_value = [(MockData(data_id, datetime.now(timezone.utc) - timedelta(days=2)), MockDatasetData(data_id, dataset_id))]
    mock_session.execute.return_value = mock_result
    
    mock_graph = AsyncMock()
    mock_graph.get_node_feedback_weights.return_value = {"node1": 1.0}
    mock_graph.get_edge_feedback_weights.return_value = {"edge1": 1.0}
    
    with patch("cognee.tasks.memify.decay_memory.get_relational_engine", return_value=mock_db), \
         patch("cognee.tasks.memify.decay_memory.get_graph_engine", return_value=mock_graph), \
         patch("cognee.tasks.memify.decay_memory.get_global_data_related_nodes", new_callable=AsyncMock) as mock_nodes, \
         patch("cognee.tasks.memify.decay_memory.get_global_data_related_edges", new_callable=AsyncMock) as mock_edges, \
         patch("cognee.tasks.memify.decay_memory.cognee.delete", new_callable=AsyncMock) as mock_delete, \
         patch("cognee.tasks.memify.decay_memory.get_default_user", new_callable=AsyncMock) as mock_user:
        
        mock_nodes.return_value = [MockNode("node1")]
        mock_edges.return_value = [MockEdge("edge1")]
        
        result = await decay_memory(dry_run=True, elapsed_hours=24.0, half_life_days=7.0, prune_threshold=0.05)
        
        assert result["status"] == "completed"
        assert result["processed_count"] == 1
        assert result["decayed_count"] == 1
        assert result["pruned_count"] == 0
        assert result["dry_run"] is True
        
        # Ensure graph engine is not called to update weights in dry run
        mock_graph.set_node_feedback_weights.assert_not_called()
        mock_graph.set_edge_feedback_weights.assert_not_called()
        mock_delete.assert_not_called()

@pytest.mark.asyncio
async def test_decay_memory_prune_threshold(mock_env):
    mock_db = MagicMock()
    mock_session = AsyncMock()
    mock_db.get_async_session.return_value.__aenter__.return_value = mock_session
    
    mock_result = MagicMock()
    data_id = uuid4()
    dataset_id = uuid4()
    mock_result.all.return_value = [(MockData(data_id, datetime.now(timezone.utc) - timedelta(days=2)), MockDatasetData(data_id, dataset_id))]
    mock_session.execute.return_value = mock_result
    
    mock_graph = AsyncMock()
    # Extremely low weight, will drop below 0.05
    mock_graph.get_node_feedback_weights.return_value = {"node1": 0.05}
    mock_graph.get_edge_feedback_weights.return_value = {"edge1": 0.05}
    
    with patch("cognee.tasks.memify.decay_memory.get_relational_engine", return_value=mock_db), \
         patch("cognee.tasks.memify.decay_memory.get_graph_engine", return_value=mock_graph), \
         patch("cognee.tasks.memify.decay_memory.get_global_data_related_nodes", new_callable=AsyncMock) as mock_nodes, \
         patch("cognee.tasks.memify.decay_memory.get_global_data_related_edges", new_callable=AsyncMock) as mock_edges, \
         patch("cognee.tasks.memify.decay_memory.cognee.delete", new_callable=AsyncMock) as mock_delete, \
         patch("cognee.tasks.memify.decay_memory.get_default_user", new_callable=AsyncMock) as mock_user:
        
        mock_nodes.return_value = [MockNode("node1")]
        mock_edges.return_value = [MockEdge("edge1")]
        
        result = await decay_memory(dry_run=False, elapsed_hours=24.0, half_life_days=7.0, prune_threshold=0.05)
        
        assert result["status"] == "completed"
        assert result["processed_count"] == 1
        assert result["decayed_count"] == 0
        assert result["pruned_count"] == 1
        
        # Verify delete was called
        mock_delete.assert_called_once()
        mock_graph.set_node_feedback_weights.assert_not_called()
