import pytest
from unittest.mock import AsyncMock, patch, MagicMock

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
            return_value=[[mock_edge]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            return_value="Generated answer",
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

        retrieved_objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects(
            query="test query", retrieved_objects=retrieved_objects
        )
        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=retrieved_objects, context=context
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

        context = await retriever.get_context_from_objects(
            query="test query", retrieved_objects=[[mock_edge]]
        )
        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=[[mock_edge]], context=context
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
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge]], [[mock_edge2]]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            side_effect=[
                "Resolved context",
                "Extended context",
                "Resolved context",
                "Extended context",
            ],  # Different contexts
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
            ],  # Query for extension, then final answer
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

        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects(
            query="test query", retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=objects, context=context
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
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge]], [[mock_edge]]],
        ) as mock_brute_force_triplet_search,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
            ],
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

        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects(
            query="test query", retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"
    # When brute_force_triplet_search returns same triplets, the loop should stop early
    assert mock_brute_force_triplet_search.call_count == 2


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
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge]], [[mock_edge]]],
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
            "cognee.modules.retrieval.graph_completion_retriever.summarize_and_generate_completion",
            return_value=("Context summary", "Generated answer"),
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
            ],  # Extension query, then final answer
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

        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects(
            query="test query", retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=objects, context=context
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

    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=1)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge]], [[mock_edge]]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            side_effect=[
                "Extension query",
                TestModel(answer="Test answer"),
            ],  # Extension query, then final answer
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

        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects(
            query="test query", retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=objects, context=context
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
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge]], [[mock_edge]]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
            ],  # Extension query, then final answer
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
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
        mock_session_user.get.return_value = None  # No user

        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects(
            query="test query", retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=objects, context=context
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
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge]], [[mock_edge]]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            return_value="Generated answer",
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

        objects = await retriever.get_retrieved_objects("test_query")
        context = await retriever.get_context_from_objects(
            query="test query", retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query="test query", retrieved_objects=objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_batch_queries_without_context(mock_edge):
    """Test get_completion batch queries retrieves context when not provided."""
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
            return_value=[[mock_edge], [mock_edge]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects(
            query_batch=["test query 1", "test query 2"]
        )
        context = await retriever.get_context_from_objects(
            query_batch=["test query 1", "test query 2"], retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query_batch=["test query 1", "test query 2"], retrieved_objects=objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 2
    assert completion[0] == "Generated answer" and completion[1] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_batch_queries_with_provided_context(mock_edge):
    """Test get_completion batch queries uses provided context."""
    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=1)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        context = await retriever.get_context_from_objects(
            query_batch=["test query 1", "test query 2"],
            retrieved_objects=[[mock_edge], [mock_edge]],
        )
        completion = await retriever.get_completion_from_context(
            query_batch=["test query 1", "test query 2"],
            retrieved_objects=[[mock_edge], [mock_edge]],
            context=context,
        )

    assert isinstance(completion, list)
    assert len(completion) == 2
    assert completion[0] == "Generated answer" and completion[1] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_batch_queries_context_extension_rounds(mock_edge):
    """Test get_completion batch queries with multiple context extension rounds."""
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
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge], [mock_edge]], [[mock_edge2], [mock_edge2]]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            side_effect=[
                "Resolved context",
                "Resolved context",
                "Extended context",
                "Extended context",
                # Final two are for the get_context_from_objects
                "Extended context",
                "Extended context",
            ],  # Different contexts
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
                # Final completions for queries:
                "Generated answer",
                "Generated answer",
            ],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects(
            query_batch=["test query 1", "test query 2"]
        )
        context = await retriever.get_context_from_objects(
            query_batch=["test query 1", "test query 2"], retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query_batch=["test query 1", "test query 2"], retrieved_objects=objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 2
    assert completion[0] == "Generated answer" and completion[1] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_batch_queries_context_extension_stops_early(mock_edge):
    """Test get_completion batch queries stops early when no new triplets found."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=4)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge], [mock_edge]], [[mock_edge], [mock_edge]]],
        ) as mock_brute_force_triplet_search,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
                # Final completions for queries:
                "Generated answer",
                "Generated answer",
            ],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects(
            query_batch=["test query 1", "test query 2"]
        )
        context = await retriever.get_context_from_objects(
            query_batch=["test query 1", "test query 2"], retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query_batch=["test query 1", "test query 2"], retrieved_objects=objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 2
    assert completion[0] == "Generated answer" and completion[1] == "Generated answer"
    # When brute_force_triplet_search returns same triplets, the loop should stop early
    assert mock_brute_force_triplet_search.call_count == 2


@pytest.mark.asyncio
async def test_get_completion_batch_queries_zero_extension_rounds(mock_edge):
    """Test get_completion batch queries with zero context extension rounds."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionContextExtensionRetriever(context_extension_rounds=0)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge], [mock_edge]]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects(
            query_batch=["test query 1", "test query 2"]
        )
        context = await retriever.get_context_from_objects(
            query_batch=["test query 1", "test query 2"], retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query_batch=["test query 1", "test query 2"], retrieved_objects=objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 2


@pytest.mark.asyncio
async def test_get_completion_batch_queries_with_response_model(mock_edge):
    """Test get_completion batch queries with custom response model."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        answer: str

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionContextExtensionRetriever(
        context_extension_rounds=1, response_model=TestModel
    )

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge], [mock_edge]], [[mock_edge], [mock_edge]]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            side_effect=[
                "Extension query",
                TestModel(answer="Test answer"),
                # Final completions for queries:
                TestModel(answer="Test answer"),
                TestModel(answer="Test answer"),
            ],  # Extension query, then final answer
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects(
            query_batch=["test query 1", "test query 2"]
        )
        context = await retriever.get_context_from_objects(
            query_batch=["test query 1", "test query 2"], retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query_batch=["test query 1", "test query 2"], retrieved_objects=objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 2
    assert isinstance(completion[0], TestModel) and isinstance(completion[1], TestModel)


@pytest.mark.asyncio
async def test_get_completion_batch_queries_duplicate_queries(mock_edge):
    """Test get_completion batch queries with duplicate queries."""
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
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge], [mock_edge]], [[mock_edge2], [mock_edge2]]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            side_effect=[
                "Resolved context",
                "Resolved context",
                "Extended context",
                "Extended context",
                # Final call to resolve text
                "Extended context",
                "Extended context",
            ],  # Different contexts
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            side_effect=[
                "Extension query",
                "Generated answer",
                # Final completions for queries:
                "Generated answer",
                "Generated answer",
            ],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects(
            query_batch=["test query 1", "test query 1"]
        )
        context = await retriever.get_context_from_objects(
            query_batch=["test query 1", "test query 1"], retrieved_objects=objects
        )
        completion = await retriever.get_completion_from_context(
            query_batch=["test query 1", "test query 1"], retrieved_objects=objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 2
