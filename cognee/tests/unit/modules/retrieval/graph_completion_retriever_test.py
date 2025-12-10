import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


@pytest.fixture
def mock_edge():
    """Create a mock edge."""
    edge = MagicMock(spec=Edge)
    return edge


@pytest.mark.asyncio
async def test_get_triplets_success(mock_edge):
    """Test successful retrieval of triplets."""
    retriever = GraphCompletionRetriever(top_k=5)

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        return_value=[mock_edge],
    ) as mock_search:
        triplets = await retriever.get_triplets("test query")

    assert len(triplets) == 1
    assert triplets[0] == mock_edge
    mock_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_triplets_empty_results():
    """Test that empty list is returned when no triplets are found."""
    retriever = GraphCompletionRetriever()

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        return_value=[],
    ):
        triplets = await retriever.get_triplets("test query")

    assert triplets == []


@pytest.mark.asyncio
async def test_get_triplets_top_k_parameter():
    """Test that top_k parameter is passed to brute_force_triplet_search."""
    retriever = GraphCompletionRetriever(top_k=10)

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        return_value=[],
    ) as mock_search:
        await retriever.get_triplets("test query")

    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["top_k"] == 10


@pytest.mark.asyncio
async def test_get_context_success(mock_edge):
    """Test successful retrieval of context."""
    retriever = GraphCompletionRetriever()

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            return_value=[mock_edge],
        ),
    ):
        context = await retriever.get_context("test query")

    assert isinstance(context, list)
    assert len(context) == 1
    assert context[0] == mock_edge


@pytest.mark.asyncio
async def test_get_context_empty_results():
    """Test that empty list is returned when no context is found."""
    retriever = GraphCompletionRetriever()

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            return_value=[],
        ),
    ):
        context = await retriever.get_context("test query")

    assert context == []


@pytest.mark.asyncio
async def test_get_context_empty_graph():
    """Test that empty list is returned when graph is empty."""
    retriever = GraphCompletionRetriever()

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=True)

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
        return_value=mock_graph_engine,
    ):
        context = await retriever.get_context("test query")

    assert context == []


@pytest.mark.asyncio
async def test_resolve_edges_to_text(mock_edge):
    """Test resolve_edges_to_text method."""
    retriever = GraphCompletionRetriever()

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
        return_value="Resolved text",
    ) as mock_resolve:
        result = await retriever.resolve_edges_to_text([mock_edge])

    assert result == "Resolved text"
    mock_resolve.assert_awaited_once_with([mock_edge])


@pytest.mark.asyncio
async def test_init_defaults():
    """Test GraphCompletionRetriever initialization with defaults."""
    retriever = GraphCompletionRetriever()

    assert retriever.top_k == 5
    assert retriever.user_prompt_path == "graph_context_for_question.txt"
    assert retriever.system_prompt_path == "answer_simple_question.txt"
    assert retriever.node_type is None
    assert retriever.node_name is None


@pytest.mark.asyncio
async def test_init_custom_params():
    """Test GraphCompletionRetriever initialization with custom parameters."""
    retriever = GraphCompletionRetriever(
        top_k=10,
        user_prompt_path="custom_user.txt",
        system_prompt_path="custom_system.txt",
    )

    assert retriever.top_k == 10
    assert retriever.user_prompt_path == "custom_user.txt"
    assert retriever.system_prompt_path == "custom_system.txt"
