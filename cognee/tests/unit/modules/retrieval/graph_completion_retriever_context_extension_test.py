import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import UUID

from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


@pytest.fixture
def mock_edge():
    """Create a mock edge."""
    edge = MagicMock(spec=Edge)
    return edge


@pytest.mark.asyncio
async def test_get_triplets_inherited(mock_edge):
    """Test that get_triplets is inherited from parent class."""
    retriever = GraphCompletionContextExtensionRetriever()

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        return_value=[mock_edge],
    ):
        triplets = await retriever.get_triplets("test query")

    assert len(triplets) == 1
    assert triplets[0] == mock_edge


@pytest.mark.asyncio
async def test_init_defaults():
    """Test GraphCompletionContextExtensionRetriever initialization with defaults."""
    retriever = GraphCompletionContextExtensionRetriever()

    assert retriever.top_k == 5
    assert retriever.user_prompt_path == "graph_context_for_question.txt"
    assert retriever.system_prompt_path == "answer_simple_question.txt"


@pytest.mark.asyncio
async def test_init_custom_params():
    """Test GraphCompletionContextExtensionRetriever initialization with custom parameters."""
    retriever = GraphCompletionContextExtensionRetriever(
        top_k=10,
        user_prompt_path="custom_user.txt",
        system_prompt_path="custom_system.txt",
        system_prompt="Custom prompt",
        node_type=str,
        node_name=["node1"],
        save_interaction=True,
        wide_search_top_k=200,
        triplet_distance_penalty=5.0,
    )

    assert retriever.top_k == 10
    assert retriever.user_prompt_path == "custom_user.txt"
    assert retriever.system_prompt_path == "custom_system.txt"
    assert retriever.system_prompt == "Custom prompt"
    assert retriever.node_type is str
    assert retriever.node_name == ["node1"]
    assert retriever.save_interaction is True
    assert retriever.wide_search_top_k == 200
    assert retriever.triplet_distance_penalty == 5.0


@pytest.mark.asyncio
async def test_get_completion_without_context(mock_edge):
    """Test get_completion retrieves context when not provided."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=1)

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
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        retrieved_objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects("test query", retrieved_objects)
        completion = await retriever.get_completion_from_context(
            "test query", retrieved_objects, context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_provided_context(mock_edge):
    """Test get_completion uses provided context."""
    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=1)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        context = await retriever.get_context_from_objects(
            "test query", retrieved_objects=[mock_edge]
        )
        completion = await retriever.get_completion_from_context(
            "test query", retrieved_objects=[mock_edge], context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_context_extension_rounds(mock_edge):
    """Test get_completion with multiple context extension rounds."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=1)

    # Create a second edge for extension rounds
    mock_edge2 = MagicMock(spec=Edge)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            retriever,
            "get_context_from_objects",
            new_callable=AsyncMock,
            side_effect=[[mock_edge], [mock_edge2]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            side_effect=["Resolved context", "Extended context"],  # Different contexts
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
            ],  # Query for extension, then final answer
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects("test query", objects)
        completion = await retriever.get_completion_from_context(
            "test query", objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_context_extension_stops_early(mock_edge):
    """Test get_completion stops early when no new triplets found."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=4)

    with (
        patch.object(
            retriever, "get_context_from_objects", new_callable=AsyncMock, return_value=[mock_edge]
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
            ],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        # When get_context returns same triplets, the loop should stop early
        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects("test query", objects)
        completion = await retriever.get_completion_from_context(
            "test query", objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_session(mock_edge):
    """Test get_completion with session caching enabled."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionContextExtensionRetriever(
        session_id="test_session", context_extension_rounds=1
    )

    mock_user = MagicMock()
    mock_user.id = "test-user-id"

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            retriever, "get_context_from_objects", new_callable=AsyncMock, return_value=[mock_edge]
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.get_conversation_history",
            return_value="Previous conversation",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.summarize_text",
            return_value="Context summary",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
            ],  # Extension query, then final answer
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.save_conversation_history",
        ) as mock_save,
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.CacheConfig"
        ) as mock_cache_config,
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.session_user"
        ) as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = mock_user

        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects("test query", objects)
        completion = await retriever.get_completion_from_context(
            "test query", objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_completion_with_save_interaction(mock_edge):
    """Test get_completion with save_interaction enabled."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)
    mock_graph_engine.add_edges = AsyncMock()

    retriever = GraphCompletionContextExtensionRetriever(
        context_extension_rounds=1, save_interaction=True
    )

    mock_node1 = MagicMock()
    mock_node2 = MagicMock()
    mock_edge.node1 = mock_node1
    mock_edge.node2 = mock_node2

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            retriever, "get_context_from_objects", new_callable=AsyncMock, return_value="mock_edge"
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
            ],  # Extension query, then final answer
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.extract_uuid_from_node",
            side_effect=[
                UUID("550e8400-e29b-41d4-a716-446655440000"),
                UUID("550e8400-e29b-41d4-a716-446655440001"),
            ],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.add_data_points",
        ) as mock_add_data,
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        context = await retriever.get_context_from_objects("test query", [mock_edge])
        completion = await retriever.get_completion_from_context(
            "test query", [mock_edge], context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    mock_add_data.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_completion_with_response_model(mock_edge):
    """Test get_completion with custom response model."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        answer: str

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=1)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            retriever, "get_context_from_objects", new_callable=AsyncMock, return_value=[mock_edge]
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.generate_completion",
            side_effect=[
                "Extension query",
                TestModel(answer="Test answer"),
            ],  # Extension query, then final answer
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects("test query", objects)
        completion = await retriever.get_completion_from_context(
            "test query", objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert isinstance(completion[0], TestModel)


@pytest.mark.asyncio
async def test_get_completion_with_session_no_user_id(mock_edge):
    """Test get_completion with session config but no user ID."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=1)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            retriever, "get_context_from_objects", new_callable=AsyncMock, return_value=[mock_edge]
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
            ],  # Extension query, then final answer
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.CacheConfig"
        ) as mock_cache_config,
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.session_user"
        ) as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = None  # No user

        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects("test query", objects)
        completion = await retriever.get_completion_from_context(
            "test query", objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1


@pytest.mark.asyncio
async def test_get_completion_zero_extension_rounds(mock_edge):
    """Test get_completion with zero context extension rounds."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=0)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            retriever, "get_context_from_objects", new_callable=AsyncMock, return_value=[mock_edge]
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_context_extension_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config
        context = await retriever.get_context_from_objects("test query", None)

    assert isinstance(context, list)
    assert len(context) == 1
