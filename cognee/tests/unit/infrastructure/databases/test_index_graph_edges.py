import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.tasks.storage.index_graph_edges import create_edge_type_datapoints, index_graph_edges


@pytest.mark.asyncio
async def test_index_graph_edges_success():
    """Test that index_graph_edges retrieves edges and delegates to index_data_points."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (
        None,
        [
            [{"relationship_name": "rel1"}, {"relationship_name": "rel1"}],
            [{"relationship_name": "rel2"}],
        ],
    )
    mock_index_data_points = AsyncMock()

    with patch.dict(
        index_graph_edges.__globals__,
        {
            "get_graph_engine": AsyncMock(return_value=mock_graph_engine),
            "index_data_points": mock_index_data_points,
        },
    ):
        await index_graph_edges()

    mock_graph_engine.get_graph_data.assert_awaited_once()
    mock_index_data_points.assert_awaited_once()

    call_args = mock_index_data_points.call_args[0][0]
    assert len(call_args) == 2
    assert all(hasattr(item, "relationship_name") for item in call_args)


@pytest.mark.asyncio
async def test_index_graph_edges_no_relationships():
    """Test that index_graph_edges handles empty relationships correctly."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (None, [])
    mock_index_data_points = AsyncMock()

    with patch.dict(
        index_graph_edges.__globals__,
        {
            "get_graph_engine": AsyncMock(return_value=mock_graph_engine),
            "index_data_points": mock_index_data_points,
        },
    ):
        await index_graph_edges()

    mock_graph_engine.get_graph_data.assert_awaited_once()
    mock_index_data_points.assert_awaited_once()

    call_args = mock_index_data_points.call_args[0][0]
    assert len(call_args) == 0


def test_create_edge_type_datapoints_uses_nonblank_edge_text():
    edges = [
        (
            "source",
            "target",
            "related_to",
            {"relationship_name": "related_to", "edge_text": " custom text "},
        )
    ]

    datapoints = create_edge_type_datapoints(edges)

    assert len(datapoints) == 1
    assert datapoints[0].relationship_name == "custom text"


def test_create_edge_type_datapoints_uses_edge_text_without_relationship_property():
    edges = [("source", "target", "related_to", {"edge_text": " custom text "})]

    datapoints = create_edge_type_datapoints(edges)

    assert len(datapoints) == 1
    assert datapoints[0].relationship_name == "custom text"


def test_create_edge_type_datapoints_falls_back_from_blank_edge_text_to_relationship_name():
    edges = [
        ("source", "target", "related_to", {"relationship_name": "related_to", "edge_text": ""}),
        ("source", "target", "mentions", {"relationship_name": "mentions", "edge_text": "   "}),
        ("source", "target", "works_at", {"relationship_name": "works_at", "edge_text": None}),
    ]

    datapoints = create_edge_type_datapoints(edges)

    relationship_names = {datapoint.relationship_name for datapoint in datapoints}
    assert relationship_names == {"related_to", "mentions", "works_at"}


def test_create_edge_type_datapoints_skips_empty_retrieval_text():
    edges = [("source", "target", "", {"relationship_name": "", "edge_text": ""})]

    assert create_edge_type_datapoints(edges) == []


@pytest.mark.asyncio
async def test_index_graph_edges_initialization_error():
    """Test that index_graph_edges raises a RuntimeError if initialization fails."""
    with patch.dict(
        index_graph_edges.__globals__,
        {
            "get_graph_engine": AsyncMock(side_effect=Exception("Graph engine failed")),
            "get_vector_engine": lambda: AsyncMock(),
        },
    ):
        with pytest.raises(RuntimeError, match="Graph edge indexing error"):
            await index_graph_edges()
