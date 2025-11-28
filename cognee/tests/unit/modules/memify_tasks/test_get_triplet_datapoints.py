import sys
import pytest
from unittest.mock import AsyncMock, patch

from cognee.tasks.memify.get_triplet_datapoints import get_triplet_datapoints
from cognee.modules.engine.models import Triplet
from cognee.modules.engine.models.Entity import Entity
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.models.EdgeType import EdgeType


get_triplet_datapoints_module = sys.modules["cognee.tasks.memify.get_triplet_datapoints"]


@pytest.fixture
def mock_graph_engine():
    """Create a mock graph engine with get_triplets_batch method."""
    engine = AsyncMock()
    engine.get_triplets_batch = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_get_triplet_datapoints_success(mock_graph_engine):
    """Test successful extraction of triplet datapoints."""
    mock_triplets_batch = [
        {
            "start_node": {
                "id": "node1",
                "type": "Entity",
                "name": "Alice",
                "description": "A person",
            },
            "end_node": {
                "id": "node2",
                "type": "Entity",
                "name": "Bob",
                "description": "Another person",
            },
            "relationship_properties": {
                "relationship_name": "knows",
            },
        }
    ]

    mock_graph_engine.get_triplets_batch.return_value = mock_triplets_batch

    with (
        patch.object(
            get_triplet_datapoints_module, "get_graph_engine", return_value=mock_graph_engine
        ),
        patch.object(get_triplet_datapoints_module, "get_all_subclasses") as mock_get_subclasses,
    ):
        mock_get_subclasses.return_value = [Triplet, EdgeType, Entity]

        triplets = []
        async for triplet in get_triplet_datapoints([{}], triplets_batch_size=100):
            triplets.append(triplet)

        assert len(triplets) == 1
        assert isinstance(triplets[0], Triplet)
        assert triplets[0].from_node_id == "node1"
        assert triplets[0].to_node_id == "node2"
        assert "Alice" in triplets[0].text
        assert "knows" in triplets[0].text
        assert "Bob" in triplets[0].text


@pytest.mark.asyncio
async def test_get_triplet_datapoints_edge_text_priority_and_fallback(mock_graph_engine):
    """Test that edge_text is prioritized over relationship_name, and fallback works."""

    class MockEntity(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    mock_triplets_batch = [
        {
            "start_node": {"id": "node1", "type": "Entity", "name": "Alice"},
            "end_node": {"id": "node2", "type": "Entity", "name": "Bob"},
            "relationship_properties": {
                "relationship_name": "knows",
                "edge_text": "has a close friendship with",
            },
        },
        {
            "start_node": {"id": "node3", "type": "Entity", "name": "Charlie"},
            "end_node": {"id": "node4", "type": "Entity", "name": "Diana"},
            "relationship_properties": {
                "relationship_name": "works_with",
            },
        },
    ]

    mock_graph_engine.get_triplets_batch.return_value = mock_triplets_batch

    with (
        patch.object(
            get_triplet_datapoints_module, "get_graph_engine", return_value=mock_graph_engine
        ),
        patch.object(get_triplet_datapoints_module, "get_all_subclasses") as mock_get_subclasses,
    ):
        mock_get_subclasses.return_value = [Triplet, EdgeType, MockEntity]

        triplets = []
        async for triplet in get_triplet_datapoints([{}], triplets_batch_size=100):
            triplets.append(triplet)

        assert len(triplets) == 2
        assert "has a close friendship with" in triplets[0].text
        assert "knows" not in triplets[0].text
        assert "works_with" in triplets[1].text


@pytest.mark.asyncio
async def test_get_triplet_datapoints_skips_missing_node_ids(mock_graph_engine):
    """Test that triplets with missing node IDs are skipped."""

    class MockEntity(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    mock_triplets_batch = [
        {
            "start_node": {"id": "", "type": "Entity", "name": "Alice"},
            "end_node": {"id": "node2", "type": "Entity", "name": "Bob"},
            "relationship_properties": {"relationship_name": "knows"},
        },
        {
            "start_node": {"id": "node3", "type": "Entity", "name": "Charlie"},
            "end_node": {"id": "node4", "type": "Entity", "name": "Diana"},
            "relationship_properties": {"relationship_name": "works_with"},
        },
    ]

    mock_graph_engine.get_triplets_batch.return_value = mock_triplets_batch

    with (
        patch.object(
            get_triplet_datapoints_module, "get_graph_engine", return_value=mock_graph_engine
        ),
        patch.object(get_triplet_datapoints_module, "get_all_subclasses") as mock_get_subclasses,
    ):
        mock_get_subclasses.return_value = [Triplet, EdgeType, MockEntity]

        triplets = []
        async for triplet in get_triplet_datapoints([{}], triplets_batch_size=100):
            triplets.append(triplet)

        assert len(triplets) == 1
        assert triplets[0].from_node_id == "node3"


@pytest.mark.asyncio
async def test_get_triplet_datapoints_error_handling(mock_graph_engine):
    """Test that errors are handled correctly - invalid data is skipped, query errors propagate."""

    class MockEntity(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    mock_triplets_batch = [
        {
            "start_node": {"id": "node1", "type": "Entity", "name": "Alice"},
            "end_node": {"id": "node2", "type": "Entity", "name": "Bob"},
            "relationship_properties": {"relationship_name": "knows"},
        },
        {
            "start_node": None,
            "end_node": {"id": "node4", "type": "Entity", "name": "Diana"},
            "relationship_properties": {"relationship_name": "works_with"},
        },
    ]

    mock_graph_engine.get_triplets_batch.return_value = mock_triplets_batch

    with (
        patch.object(
            get_triplet_datapoints_module, "get_graph_engine", return_value=mock_graph_engine
        ),
        patch.object(get_triplet_datapoints_module, "get_all_subclasses") as mock_get_subclasses,
    ):
        mock_get_subclasses.return_value = [Triplet, EdgeType, MockEntity]

        triplets = []
        async for triplet in get_triplet_datapoints([{}], triplets_batch_size=100):
            triplets.append(triplet)

        assert len(triplets) == 1
        assert triplets[0].from_node_id == "node1"

    mock_graph_engine.get_triplets_batch.side_effect = Exception("Database connection error")

    with patch.object(
        get_triplet_datapoints_module, "get_graph_engine", return_value=mock_graph_engine
    ):
        triplets = []
        with pytest.raises(Exception, match="Database connection error"):
            async for triplet in get_triplet_datapoints([{}], triplets_batch_size=100):
                triplets.append(triplet)


@pytest.mark.asyncio
async def test_get_triplet_datapoints_no_get_triplets_batch_method(mock_graph_engine):
    """Test that NotImplementedError is raised when graph engine lacks get_triplets_batch."""
    del mock_graph_engine.get_triplets_batch

    with patch.object(
        get_triplet_datapoints_module, "get_graph_engine", return_value=mock_graph_engine
    ):
        triplets = []
        with pytest.raises(NotImplementedError, match="does not support get_triplets_batch"):
            async for triplet in get_triplet_datapoints([{}], triplets_batch_size=100):
                triplets.append(triplet)
