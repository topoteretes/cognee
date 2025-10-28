import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.tasks.storage.index_graph_edges import index_graph_edges


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
        with pytest.raises(RuntimeError, match="Initialization error"):
            await index_graph_edges()
