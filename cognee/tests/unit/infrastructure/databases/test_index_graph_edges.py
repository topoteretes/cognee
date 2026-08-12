import pytest
from unittest.mock import AsyncMock, patch

from cognee.modules.graph.models.EdgeInstance import EdgeInstance
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.engine.utils import generate_edge_object_id
from cognee.tasks.storage.index_graph_edges import index_graph_edges


@pytest.mark.asyncio
async def test_edge_index_contract_standard_path():
    """Structural edge types use graph counts while prose stays on edge instances."""
    graph = AsyncMock()
    graph.get_edge_type_counts.return_value = {"depends_on": 1}
    vector = AsyncMock()
    indexer = AsyncMock()
    edges = [
        (
            "source",
            "target",
            "depends_on",
            {
                "edge_text": "Source depends on Target because the build requires it.",
            },
        )
    ]

    with patch.dict(index_graph_edges.__globals__, {"index_data_points": indexer}):
        await index_graph_edges(edges, vector_engine=vector, graph_engine=graph)

    points = indexer.await_args.args[0]
    edge_type = next(point for point in points if isinstance(point, EdgeType))
    edge_instance = next(point for point in points if isinstance(point, EdgeInstance))
    type_id = str(edge_type.id)
    instance_id = str(edge_instance.id)
    type_text = edge_type.relationship_name
    instance_text = edge_instance.text

    assert type_id == str(EdgeType.id_for("depends_on"))
    assert instance_id == generate_edge_object_id("source", "target", "depends_on")
    assert type_text == "depends_on"
    assert instance_text == "Source depends on Target because the build requires it."
    assert edge_type.number_of_edges == 1
    graph.get_edge_type_counts.assert_awaited_once_with(["depends_on"])


@pytest.mark.asyncio
async def test_index_graph_edges_initialization_error():
    """Test that index_graph_edges raises a RuntimeError if initialization fails."""
    with patch.dict(
        index_graph_edges.__globals__,
        {
            "get_graph_engine": AsyncMock(side_effect=Exception("Graph engine failed")),
            "get_vector_engine_async": AsyncMock(return_value=AsyncMock()),
        },
    ):
        with pytest.raises(RuntimeError, match="Graph edge indexing error"):
            await index_graph_edges()
