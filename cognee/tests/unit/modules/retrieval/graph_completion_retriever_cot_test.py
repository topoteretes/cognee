import os

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import UUID
from itertools import cycle

from cognee.modules.retrieval.graph_completion_cot_retriever import (
    GraphCompletionCotRetriever,
    _as_answer_text,
)
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.infrastructure.llm.LLMGateway import LLMGateway


@pytest.fixture
def mock_edge():
    """Create a mock edge."""
    edge = MagicMock(spec=Edge)
    return edge


@pytest.mark.asyncio
async def test_get_triplets_inherited(mock_edge):
    """Test that get_triplets is inherited from parent class."""
    retriever = GraphCompletionCotRetriever()

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        return_value=[mock_edge],
    ):
        triplets = await retriever.get_triplets("test query")

    assert len(triplets) == 1
    assert triplets[0] == mock_edge


@pytest.mark.asyncio
async def test_init_custom_params():
    """Test GraphCompletionCotRetriever initialization with custom parameters."""
    retriever = GraphCompletionCotRetriever(
        top_k=10,
        user_prompt_path="custom_user.txt",
        system_prompt_path="custom_system.txt",
        validation_user_prompt_path="custom_validation_user.txt",
        validation_system_prompt_path="custom_validation_system.txt",
        followup_system_prompt_path="custom_followup_system.txt",
        followup_user_prompt_path="custom_followup_user.txt",
    )

    assert retriever.top_k == 10
    assert retriever.user_prompt_path == "custom_user.txt"
    assert retriever.system_prompt_path == "custom_system.txt"
    assert retriever.validation_user_prompt_path == "custom_validation_user.txt"
    assert retriever.validation_system_prompt_path == "custom_validation_system.txt"
    assert retriever.followup_system_prompt_path == "custom_followup_system.txt"
    assert retriever.followup_user_prompt_path == "custom_followup_user.txt"


@pytest.mark.asyncio
async def test_init_defaults():
    """Test GraphCompletionCotRetriever initialization with defaults."""
    retriever = GraphCompletionCotRetriever()

    assert retriever.validation_user_prompt_path == "cot_validation_user_prompt.txt"
    assert retriever.validation_system_prompt_path == "cot_validation_system_prompt.txt"
    assert retriever.followup_system_prompt_path == "cot_followup_system_prompt.txt"
    assert retriever.followup_user_prompt_path == "cot_followup_user_prompt.txt"


@pytest.mark.asyncio
async def test_run_cot_completion_round_zero_with_context(mock_edge):
    """Test _run_cot_completion round 0 with provided context."""
    retriever = GraphCompletionCotRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch.object(retriever, "get_context", new_callable=AsyncMock, return_value=[[mock_edge]]),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever._as_answer_text",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.render_prompt",
            return_value="Rendered prompt",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.read_query_prompt",
            return_value="System prompt",
        ),
        patch.object(
            LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            side_effect=["validation_result", "followup_question"],
        ),
    ):
        completion, context_text, triplets = await retriever._run_cot_completion(
            query="test query",
            context=[mock_edge],
            max_iter=1,
        )

    assert completion == ["Generated answer"]
    assert context_text == ["Resolved context"]
    assert len(triplets) >= 1


@pytest.mark.asyncio
async def test_run_cot_completion_round_zero_without_context(mock_edge):
    """Test _run_cot_completion round 0 without provided context."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionCotRetriever()

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
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
    ):
        completion, context_text, triplets = await retriever._run_cot_completion(
            query="test query",
            context=None,
            max_iter=1,
        )

    assert completion == ["Generated answer"]
    assert context_text == ["Resolved context"]
    assert len(triplets) >= 1


@pytest.mark.asyncio
async def test_run_cot_completion_multiple_rounds(mock_edge):
    """Test _run_cot_completion with multiple rounds."""
    retriever = GraphCompletionCotRetriever()

    mock_edge2 = MagicMock(spec=Edge)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch.object(
            retriever,
            "get_context",
            new_callable=AsyncMock,
            side_effect=[[[mock_edge]], [[mock_edge2]]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.render_prompt",
            return_value="Rendered prompt",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.read_query_prompt",
            return_value="System prompt",
        ),
        patch.object(
            LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            side_effect=[
                "validation_result",
                "followup_question",
                "validation_result2",
                "followup_question2",
            ],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever._as_answer_text",
            return_value="Generated answer",
        ),
    ):
        completion, context_text, triplets = await retriever._run_cot_completion(
            query="test query",
            context=[mock_edge],
            max_iter=2,
        )

    assert completion == ["Generated answer"]
    assert context_text == ["Resolved context"]
    assert len(triplets) >= 1


@pytest.mark.asyncio
async def test_run_cot_completion_with_conversation_history(mock_edge):
    """Test _run_cot_completion with conversation history."""
    retriever = GraphCompletionCotRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ) as mock_generate,
    ):
        completion, context_text, triplets = await retriever._run_cot_completion(
            query="test query",
            context=[mock_edge],
            conversation_history="Previous conversation",
            max_iter=1,
        )

    assert completion == ["Generated answer"]
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs.get("conversation_history") == "Previous conversation"


@pytest.mark.asyncio
async def test_run_cot_completion_with_response_model(mock_edge):
    """Test _run_cot_completion with custom response model."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        answer: str

    retriever = GraphCompletionCotRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value=TestModel(answer="Test answer"),
        ),
    ):
        completion, context_text, triplets = await retriever._run_cot_completion(
            query="test query",
            context=[mock_edge],
            response_model=TestModel,
            max_iter=1,
        )

    assert isinstance(completion, list)
    assert isinstance(completion[0], TestModel)
    assert completion[0].answer == "Test answer"


@pytest.mark.asyncio
async def test_run_cot_completion_empty_conversation_history(mock_edge):
    """Test _run_cot_completion with empty conversation history."""
    retriever = GraphCompletionCotRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ) as mock_generate,
    ):
        completion, context_text, triplets = await retriever._run_cot_completion(
            query="test query",
            context=[mock_edge],
            conversation_history="",
            max_iter=1,
        )

    assert completion == ["Generated answer"]
    # Verify conversation_history was passed as None when empty
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs.get("conversation_history") is None


@pytest.mark.asyncio
async def test_get_completion_without_context(mock_edge):
    """Test get_completion retrieves context when not provided."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionCotRetriever()

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
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch.object(retriever, "get_context", new_callable=AsyncMock, return_value=[[mock_edge]]),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever._as_answer_text",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.render_prompt",
            return_value="Rendered prompt",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.read_query_prompt",
            return_value="System prompt",
        ),
        patch.object(
            LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            side_effect=["validation_result", "followup_question"],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion("test query", max_iter=1)

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_provided_context(mock_edge):
    """Test get_completion uses provided context."""
    retriever = GraphCompletionCotRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion("test query", context=[[mock_edge]], max_iter=1)

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_session(mock_edge):
    """Test get_completion with session caching enabled."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionCotRetriever()

    mock_user = MagicMock()
    mock_user.id = "test-user-id"

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
            "cognee.modules.retrieval.graph_completion_cot_retriever.get_conversation_history",
            return_value="Previous conversation",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.summarize_text",
            return_value="Context summary",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.save_conversation_history",
        ) as mock_save,
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.CacheConfig"
        ) as mock_cache_config,
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.session_user"
        ) as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = mock_user

        completion = await retriever.get_completion(
            "test query", session_id="test_session", max_iter=1
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

    retriever = GraphCompletionCotRetriever(save_interaction=True)

    mock_node1 = MagicMock()
    mock_node2 = MagicMock()
    mock_edge.node1 = mock_node1
    mock_edge.node2 = mock_node2

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch.object(retriever, "get_context", new_callable=AsyncMock, return_value=[[mock_edge]]),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever._as_answer_text",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.render_prompt",
            return_value="Rendered prompt",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.read_query_prompt",
            return_value="System prompt",
        ),
        patch.object(
            LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            side_effect=["validation_result", "followup_question"],
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
            "cognee.modules.retrieval.graph_completion_cot_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        # Pass context so save_interaction condition is met
        completion = await retriever.get_completion("test query", context=[mock_edge], max_iter=1)

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

    retriever = GraphCompletionCotRetriever()

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
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value=TestModel(answer="Test answer"),
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion(
            "test query", response_model=TestModel, max_iter=1
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert isinstance(completion[0], TestModel)


@pytest.mark.asyncio
async def test_get_completion_with_session_no_user_id(mock_edge):
    """Test get_completion with session config but no user ID."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionCotRetriever()

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
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.CacheConfig"
        ) as mock_cache_config,
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.session_user"
        ) as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = None  # No user

        completion = await retriever.get_completion("test query", max_iter=1)

    assert isinstance(completion, list)
    assert len(completion) == 1


@pytest.mark.asyncio
async def test_get_completion_with_save_interaction_no_context(mock_edge):
    """Test get_completion with save_interaction but no context provided."""
    retriever = GraphCompletionCotRetriever(save_interaction=True)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch.object(retriever, "get_context", new_callable=AsyncMock, return_value=[[mock_edge]]),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever._as_answer_text",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.render_prompt",
            return_value="Rendered prompt",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.read_query_prompt",
            return_value="System prompt",
        ),
        patch.object(
            LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            side_effect=["validation_result", "followup_question"],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion("test query", context=None, max_iter=1)

    assert isinstance(completion, list)
    assert len(completion) == 1


@pytest.mark.asyncio
async def test_as_answer_text_with_typeerror():
    """Test _as_answer_text handles TypeError when json.dumps fails."""
    non_serializable = {1, 2, 3}

    result = _as_answer_text(non_serializable)

    assert isinstance(result, str)
    assert result == str(non_serializable)


@pytest.mark.asyncio
async def test_as_answer_text_with_string():
    """Test _as_answer_text with string input."""
    result = _as_answer_text("test string")
    assert result == "test string"


@pytest.mark.asyncio
async def test_as_answer_text_with_dict():
    """Test _as_answer_text with dictionary input."""
    test_dict = {"key": "value", "number": 42}
    result = _as_answer_text(test_dict)
    assert isinstance(result, str)
    assert "key" in result
    assert "value" in result


@pytest.mark.asyncio
async def test_as_answer_text_with_basemodel():
    """Test _as_answer_text with Pydantic BaseModel input."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        answer: str

    test_model = TestModel(answer="test answer")
    result = _as_answer_text(test_model)

    assert isinstance(result, str)
    assert "[Structured Response]" in result
    assert "test answer" in result


@pytest.mark.asyncio
async def test_get_completion_batch_queries_with_context(mock_edge):
    """Test get_completion batch queries with provided context."""
    retriever = GraphCompletionCotRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            return_value="Resolved context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch.object(retriever, "get_context", new_callable=AsyncMock, return_value=[[mock_edge]]),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever._as_answer_text",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.render_prompt",
            return_value="Rendered prompt",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.read_query_prompt",
            return_value="System prompt",
        ),
        patch.object(
            LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            side_effect=cycle(["validation_result", "followup_question"]),
        ),
    ):
        completion = await retriever.get_completion(
            query_batch=["test query 1", "test query 2"],
            context=[[mock_edge], [mock_edge]],
            max_iter=1,
        )

    assert isinstance(completion, list)
    assert len(completion) == 2
    assert completion[0] == "Generated answer" and completion[1] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_batch_queries_without_context(mock_edge):
    """Test get_completion batch queries without provided context."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionCotRetriever()

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
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value="Generated answer",
        ),
    ):
        completion = await retriever.get_completion(
            query_batch=["test query 1", "test query 2"], max_iter=1
        )

    assert isinstance(completion, list)
    assert len(completion) == 2
    assert completion[0] == "Generated answer" and completion[1] == "Generated answer"


#
@pytest.mark.asyncio
async def test_get_completion_batch_queries_with_response_model(mock_edge):
    """Test get_completion of batch queries with custom response model."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        answer: str

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    retriever = GraphCompletionCotRetriever()

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
            "cognee.modules.retrieval.graph_completion_cot_retriever.generate_completion",
            return_value=TestModel(answer="Test answer"),
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_cot_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion(
            query_batch=["test query 1", "test query 2"], response_model=TestModel, max_iter=1
        )

        assert isinstance(completion, list)
        assert len(completion) == 2
        assert isinstance(completion[0], TestModel) and isinstance(completion[1], TestModel)
