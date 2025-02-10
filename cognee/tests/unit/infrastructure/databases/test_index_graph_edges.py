import pytest
from unittest.mock import AsyncMock, patch
from cognee.tasks.storage.index_graph_edges import index_graph_edges


@pytest.mark.asyncio
async def test_index_graph_edges_success():
    """Test that index_graph_edges uses the index datapoints and creates vector index."""
    # Create the mocks for the graph and vector engines.
    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (
        None,
        [
            [{"relationship_name": "rel1"}, {"relationship_name": "rel1"}],
            [{"relationship_name": "rel2"}],
        ],
    )
    mock_vector_engine = AsyncMock()

    # Patch the globals of the function so that when it does:
    #   vector_engine = get_vector_engine()
    #   graph_engine = await get_graph_engine()
    # it uses the mocked versions.
    with patch.dict(
        index_graph_edges.__globals__,
        {
            "get_graph_engine": AsyncMock(return_value=mock_graph_engine),
            "get_vector_engine": lambda: mock_vector_engine,
        },
    ):
        await index_graph_edges()

    # Assertions on the mock calls.
    mock_graph_engine.get_graph_data.assert_awaited_once()
    assert mock_vector_engine.create_vector_index.await_count == 1
    assert mock_vector_engine.index_data_points.await_count == 1


@pytest.mark.asyncio
async def test_index_graph_edges_no_relationships():
    """Test that index_graph_edges handles empty relationships correctly."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (None, [])
    mock_vector_engine = AsyncMock()

    with patch.dict(
        index_graph_edges.__globals__,
        {
            "get_graph_engine": AsyncMock(return_value=mock_graph_engine),
            "get_vector_engine": lambda: mock_vector_engine,
        },
    ):
        await index_graph_edges()

    mock_graph_engine.get_graph_data.assert_awaited_once()
    mock_vector_engine.create_vector_index.assert_not_awaited()
    mock_vector_engine.index_data_points.assert_not_awaited()


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
