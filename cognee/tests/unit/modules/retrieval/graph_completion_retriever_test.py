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

    mock_node1 = MagicMock()
    mock_node2 = MagicMock()
    mock_edge.node1 = mock_node1
    mock_edge.node2 = mock_node2
    mock_edge.attributes = {"text": "mock edge"}

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
        context = await retriever.get_context_from_objects(
            query="test query", retrieved_objects=[mock_edge]
        )

    assert isinstance(context, str)


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
        context = await retriever.get_context_from_objects(query="test query", retrieved_objects=[])

    assert context == ""


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
        context = await retriever.get_context_from_objects(query="test query", retrieved_objects=[])

    assert context == ""


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
        system_prompt="Custom prompt",
        node_type=str,
        node_name=["node1"],
        wide_search_top_k=200,
        triplet_distance_penalty=5.0,
    )

    assert retriever.top_k == 10
    assert retriever.user_prompt_path == "custom_user.txt"
    assert retriever.system_prompt_path == "custom_system.txt"
    assert retriever.system_prompt == "Custom prompt"
    assert retriever.node_type is str
    assert retriever.node_name == ["node1"]
    assert retriever.wide_search_top_k == 200
    assert retriever.triplet_distance_penalty == 5.0


@pytest.mark.asyncio
async def test_init_none_top_k():
    """Test GraphCompletionRetriever initialization with None top_k."""
    retriever = GraphCompletionRetriever(top_k=None)

    assert retriever.top_k == 5  # None defaults to 5


@pytest.mark.asyncio
async def test_get_completion_without_context(mock_edge):
    """Test get_completion retrieves context when not provided."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            return_value=[mock_edge],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=None, context=None
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_provided_context(mock_edge):
    """Test get_completion uses provided context."""
    retriever = GraphCompletionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=None, context="mock edge"
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_session(mock_edge):
    """Test get_completion with session caching enabled."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionRetriever(session_id="test_session")

    mock_user = MagicMock()
    mock_user.id = "test-user-id"

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            return_value=[mock_edge],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_conversation_history",
            return_value="Previous conversation",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.summarize_text",
            return_value="Context summary",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.save_conversation_history",
        ) as mock_save,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.session_user"
        ) as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = mock_user

        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=None, context=None
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_completion_with_response_model(mock_edge):
    """Test get_completion with custom response model."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        answer: str

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionRetriever(response_model=TestModel)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            return_value=[mock_edge],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.generate_completion",
            return_value=TestModel(answer="Test answer"),
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=None, context=None
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert isinstance(completion[0], TestModel)


@pytest.mark.asyncio
async def test_get_completion_empty_context(mock_edge):
    """Test get_completion with empty context."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            return_value=[],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=None, context=None
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
