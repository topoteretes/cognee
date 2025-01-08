import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_index_graph_edges_success():
    """Test that index_graph_edges uses the index datapoints and creates vector index."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (
        None,
        [
            [{"relationship_name": "rel1"}, {"relationship_name": "rel1"}],
            [{"relationship_name": "rel2"}],
        ],
    )

    mock_vector_engine = AsyncMock()

    with (
        patch(
            "cognee.tasks.storage.index_graph_edges.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.tasks.storage.index_graph_edges.get_vector_engine",
            return_value=mock_vector_engine,
        ),
    ):
        from cognee.tasks.storage.index_graph_edges import index_graph_edges

        await index_graph_edges()

        mock_graph_engine.get_graph_data.assert_awaited_once()
        assert mock_vector_engine.create_vector_index.await_count == 1
        assert mock_vector_engine.index_data_points.await_count == 1


@pytest.mark.asyncio
async def test_index_graph_edges_no_relationships():
    """Test that index_graph_edges handles empty relationships correctly."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (None, [])

    mock_vector_engine = AsyncMock()

    with (
        patch(
            "cognee.tasks.storage.index_graph_edges.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.tasks.storage.index_graph_edges.get_vector_engine",
            return_value=mock_vector_engine,
        ),
    ):
        from cognee.tasks.storage.index_graph_edges import index_graph_edges

        await index_graph_edges()

        mock_graph_engine.get_graph_data.assert_awaited_once()
        mock_vector_engine.create_vector_index.assert_not_awaited()
        mock_vector_engine.index_data_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_index_graph_edges_initialization_error():
    """Test that index_graph_edges raises a RuntimeError if initialization fails."""
    with (
        patch(
            "cognee.tasks.storage.index_graph_edges.get_graph_engine",
            side_effect=Exception("Graph engine failed"),
        ),
        patch("cognee.tasks.storage.index_graph_edges.get_vector_engine", return_value=AsyncMock()),
    ):
        from cognee.tasks.storage.index_graph_edges import index_graph_edges

        with pytest.raises(RuntimeError, match="Initialization error"):
            await index_graph_edges()
