import pytest
from unittest.mock import AsyncMock, patch

from cognee.modules.graph.models.EdgeInstance import EdgeInstance
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.tasks.storage.index_graph_edges import index_graph_edges


@pytest.mark.asyncio
async def test_index_graph_edges_indexes_types_and_instances_with_graph_counts():
    """Structural edge types use graph counts while prose stays on edge instances."""
    graph = AsyncMock()
    graph.get_edge_type_counts.return_value = {"depends_on": 7}
    vector = AsyncMock()
    indexer = AsyncMock()
    edges = [
        (
            "a",
            "b",
            "depends_on",
            {
                "edge_object_id": "15bfc0f0-51d7-5ac8-8589-3c32fe75aa10",
                "edge_text": "A depends on B.",
            },
        )
    ]

    with patch.dict(index_graph_edges.__globals__, {"index_data_points": indexer}):
        await index_graph_edges(edges, vector_engine=vector, graph_engine=graph)

    points = indexer.await_args.args[0]
    assert [(p.relationship_name, p.number_of_edges) for p in points if isinstance(p, EdgeType)] == [
        ("depends_on", 7)
    ]
    assert [p.text for p in points if isinstance(p, EdgeInstance)] == ["A depends on B."]
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
