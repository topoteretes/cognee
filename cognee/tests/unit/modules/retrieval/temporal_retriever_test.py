from types import SimpleNamespace
import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from cognee.modules.retrieval.temporal_retriever import TemporalRetriever
from cognee.tasks.temporal_graph.models import QueryInterval, Timestamp
from cognee.infrastructure.llm import LLMGateway


# Test TemporalRetriever initialization defaults and overrides
def test_init_defaults_and_overrides():
    tr = TemporalRetriever()
    assert tr.top_k == 5
    assert tr.user_prompt_path == "graph_context_for_question.txt"
    assert tr.system_prompt_path == "answer_simple_question.txt"
    assert tr.time_extraction_prompt_path == "extract_query_time.txt"

    tr2 = TemporalRetriever(
        top_k=3,
        user_prompt_path="u.txt",
        system_prompt_path="s.txt",
        time_extraction_prompt_path="t.txt",
    )
    assert tr2.top_k == 3
    assert tr2.user_prompt_path == "u.txt"
    assert tr2.system_prompt_path == "s.txt"
    assert tr2.time_extraction_prompt_path == "t.txt"


# Test descriptions_to_string with basic and empty results
def test_descriptions_to_string_basic_and_empty():
    tr = TemporalRetriever()

    results = [
        {"description": "  First  "},
        {"nope": "no description"},
        {"description": "Second"},
        {"description": ""},
        {"description": "   Third line   "},
    ]

    s = tr.descriptions_to_string(results)
    assert s == "First\n#####################\nSecond\n#####################\nThird line"

    assert tr.descriptions_to_string([]) == ""


# Test filter_top_k_events sorts and limits correctly
@pytest.mark.asyncio
async def test_filter_top_k_events_sorts_and_limits():
    tr = TemporalRetriever(top_k=2)

    relevant_events = [
        {
            "events": [
                {"id": "e1", "description": "E1"},
                {"id": "e2", "description": "E2"},
                {"id": "e3", "description": "E3 - not in vector results"},
            ]
        }
    ]

    scored_results = [
        SimpleNamespace(id="e2", payload={"id": "e2"}, score=0.10),
        SimpleNamespace(id="e1", payload={"id": "e1"}, score=0.20),
    ]

    top = await tr.filter_top_k_events(relevant_events, scored_results)

    assert [e["id"] for e in top] == ["e2", "e1"]
    assert all("score" in e for e in top)
    assert top[0]["score"] == 0.10
    assert top[1]["score"] == 0.20


# Test filter_top_k_events handles unknown ids as infinite scores
@pytest.mark.asyncio
async def test_filter_top_k_events_includes_unknown_as_infinite_but_not_in_top_k():
    tr = TemporalRetriever(top_k=2)

    relevant_events = [
        {
            "events": [
                {"id": "known1", "description": "Known 1"},
                {"id": "unknown", "description": "Unknown"},
                {"id": "known2", "description": "Known 2"},
            ]
        }
    ]

    scored_results = [
        SimpleNamespace(id="known2", payload={"id": "known2"}, score=0.05),
        SimpleNamespace(id="known1", payload={"id": "known1"}, score=0.50),
    ]

    top = await tr.filter_top_k_events(relevant_events, scored_results)
    assert [e["id"] for e in top] == ["known2", "known1"]
    assert all(e["score"] != float("inf") for e in top)


# Test descriptions_to_string with unicode and newlines
def test_descriptions_to_string_unicode_and_newlines():
    tr = TemporalRetriever()
    results = [
        {"description": "Line A\nwith newline"},
        {"description": "This is a description"},
    ]
    s = tr.descriptions_to_string(results)
    assert "Line A\nwith newline" in s
    assert "This is a description" in s
    assert s.count("#####################") == 1


# Test filter_top_k_events when top_k is larger than available events
@pytest.mark.asyncio
async def test_filter_top_k_events_limits_when_top_k_exceeds_events():
    tr = TemporalRetriever(top_k=10)
    relevant_events = [{"events": [{"id": "a"}, {"id": "b"}]}]
    scored_results = [
        SimpleNamespace(id="a", payload={"id": "a"}, score=0.1),
        SimpleNamespace(id="b", payload={"id": "b"}, score=0.2),
    ]
    out = await tr.filter_top_k_events(relevant_events, scored_results)
    assert [e["id"] for e in out] == ["a", "b"]


# Test filter_top_k_events when scored_results is empty
@pytest.mark.asyncio
async def test_filter_top_k_events_handles_empty_scored_results():
    tr = TemporalRetriever(top_k=2)
    relevant_events = [{"events": [{"id": "x"}, {"id": "y"}]}]
    scored_results = []
    out = await tr.filter_top_k_events(relevant_events, scored_results)
    assert [e["id"] for e in out] == ["x", "y"]
    assert all(e["score"] == float("inf") for e in out)


# Test filter_top_k_events error handling for missing structure
@pytest.mark.asyncio
async def test_filter_top_k_events_error_handling():
    tr = TemporalRetriever(top_k=2)
    with pytest.raises((KeyError, TypeError)):
        await tr.filter_top_k_events([{}], [])


@pytest.fixture
def mock_graph_engine():
    """Create a mock graph engine."""
    engine = AsyncMock()
    engine.collect_time_ids = AsyncMock()
    engine.collect_events = AsyncMock()
    return engine


@pytest.fixture
def mock_vector_engine():
    """Create a mock vector engine."""
    engine = AsyncMock()
    engine.embedding_engine = AsyncMock()
    engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    engine.search = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_get_context_with_time_range(mock_graph_engine, mock_vector_engine):
    """Test get_context when time range is extracted from query."""
    retriever = TemporalRetriever(top_k=5)

    mock_graph_engine.collect_time_ids.return_value = ["e1", "e2"]
    mock_graph_engine.collect_events.return_value = [
        {
            "events": [
                {"id": "e1", "description": "Event 1"},
                {"id": "e2", "description": "Event 2"},
            ]
        }
    ]

    mock_result1 = SimpleNamespace(id="e2", payload={"id": "e2"}, score=0.05)
    mock_result2 = SimpleNamespace(id="e1", payload={"id": "e1"}, score=0.10)
    mock_vector_engine.search.return_value = [mock_result1, mock_result2]

    with (
        patch.object(
            retriever, "extract_time_from_query", return_value=("2024-01-01", "2024-12-31")
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
    ):
        objects = await retriever.get_retrieved_objects("What happened in 2024?")
        context = await retriever.get_context_from_objects("What happened in 2024?", objects)

    assert isinstance(context, str)
    assert len(context) > 0
    assert "Event" in context


@pytest.mark.asyncio
async def test_get_context_fallback_to_triplets_no_time(mock_graph_engine):
    """Test get_context falls back to triplets when no time is extracted."""
    retriever = TemporalRetriever()

    with (
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            retriever, "get_triplets", return_value=[{"s": "a", "p": "b", "o": "c"}]
        ) as mock_get_triplets,
        patch.object(
            retriever, "resolve_edges_to_text", return_value="triplet text"
        ) as mock_resolve,
    ):

        async def mock_extract_time(query):
            return None, None

        retriever.extract_time_from_query = mock_extract_time

        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", objects)

    assert context == "triplet text"
    mock_get_triplets.assert_awaited_once_with("test query")
    mock_resolve.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_context_no_events_found(mock_graph_engine):
    """Test get_context falls back to triplets when no events are found."""
    retriever = TemporalRetriever()

    mock_graph_engine.collect_time_ids.return_value = []

    with (
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            retriever, "get_triplets", return_value=[{"s": "a", "p": "b", "o": "c"}]
        ) as mock_get_triplets,
        patch.object(
            retriever, "resolve_edges_to_text", return_value="triplet text"
        ) as mock_resolve,
    ):

        async def mock_extract_time(query):
            return "2024-01-01", "2024-12-31"

        retriever.extract_time_from_query = mock_extract_time

        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", objects)

    assert context == "triplet text"
    mock_get_triplets.assert_awaited_once_with("test query")
    mock_resolve.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_context_time_from_only(mock_graph_engine, mock_vector_engine):
    """Test get_context with only time_from."""
    retriever = TemporalRetriever(top_k=5)

    mock_graph_engine.collect_time_ids.return_value = ["e1"]
    mock_graph_engine.collect_events.return_value = [
        {
            "events": [
                {"id": "e1", "description": "Event 1"},
            ]
        }
    ]

    mock_result = SimpleNamespace(id="e1", payload={"id": "e1"}, score=0.05)
    mock_vector_engine.search.return_value = [mock_result]

    with (
        patch.object(retriever, "extract_time_from_query", return_value=("2024-01-01", None)),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
    ):
        objects = await retriever.get_retrieved_objects("What happened in 2024?")
        context = await retriever.get_context_from_objects("What happened in 2024?", objects)

    assert isinstance(context, str)
    assert "Event 1" in context


@pytest.mark.asyncio
async def test_get_context_time_to_only(mock_graph_engine, mock_vector_engine):
    """Test get_context with only time_to."""
    retriever = TemporalRetriever(top_k=5)

    mock_graph_engine.collect_time_ids.return_value = ["e1"]
    mock_graph_engine.collect_events.return_value = [
        {
            "events": [
                {"id": "e1", "description": "Event 1"},
            ]
        }
    ]

    mock_result = SimpleNamespace(id="e1", payload={"id": "e1"}, score=0.05)
    mock_vector_engine.search.return_value = [mock_result]

    with (
        patch.object(retriever, "extract_time_from_query", return_value=(None, "2024-12-31")),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
    ):
        objects = await retriever.get_retrieved_objects("What happened in 2024?")
        context = await retriever.get_context_from_objects("What happened in 2024?", objects)

    assert isinstance(context, str)
    assert "Event 1" in context


@pytest.mark.asyncio
async def test_get_completion_without_context(mock_graph_engine, mock_vector_engine):
    """Test get_completion retrieves context when not provided."""
    retriever = TemporalRetriever()

    mock_graph_engine.collect_time_ids.return_value = ["e1"]
    mock_graph_engine.collect_events.return_value = [
        {
            "events": [
                {"id": "e1", "description": "Event 1"},
            ]
        }
    ]

    mock_result = SimpleNamespace(id="e1", payload={"id": "e1"}, score=0.05)
    mock_vector_engine.search.return_value = [mock_result]

    with (
        patch.object(
            retriever, "extract_time_from_query", return_value=("2024-01-01", "2024-12-31")
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.temporal_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects("What happened in 2024?")
        context = await retriever.get_context_from_objects("What happened in 2024?", objects)
        completion = await retriever.get_completion_from_context(
            "What happened in 2024?", objects, context=context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_provided_context():
    """Test get_completion uses provided context."""
    retriever = TemporalRetriever()

    with (
        patch(
            "cognee.modules.retrieval.temporal_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.temporal_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects("What happened in 2024?")
        await retriever.get_context_from_objects("What happened in 2024?", objects)
        completion = await retriever.get_completion_from_context(
            "test query", objects, context="Provided context"
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_session(mock_graph_engine, mock_vector_engine):
    """Test get_completion with session caching enabled."""
    retriever = TemporalRetriever(session_id="test_session")

    mock_graph_engine.collect_time_ids.return_value = ["e1"]
    mock_graph_engine.collect_events.return_value = [
        {
            "events": [
                {"id": "e1", "description": "Event 1"},
            ]
        }
    ]

    mock_result = SimpleNamespace(id="e1", payload={"id": "e1"}, score=0.05)
    mock_vector_engine.search.return_value = [mock_result]

    mock_user = MagicMock()
    mock_user.id = "test-user-id"

    with (
        patch.object(
            retriever, "extract_time_from_query", return_value=("2024-01-01", "2024-12-31")
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_conversation_history",
            return_value="Previous conversation",
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.summarize_text",
            return_value="Context summary",
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.save_conversation_history",
        ) as mock_save,
        patch("cognee.modules.retrieval.temporal_retriever.CacheConfig") as mock_cache_config,
        patch("cognee.modules.retrieval.temporal_retriever.session_user") as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = mock_user

        objects = await retriever.get_retrieved_objects("What happened in 2024?")
        context = await retriever.get_context_from_objects("What happened in 2024?", objects)
        completion = await retriever.get_completion_from_context(
            "What happened in 2024?", objects, context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_completion_with_session_no_user_id(mock_graph_engine, mock_vector_engine):
    """Test get_completion with session config but no user ID."""
    retriever = TemporalRetriever()

    mock_graph_engine.collect_time_ids.return_value = ["e1"]
    mock_graph_engine.collect_events.return_value = [
        {
            "events": [
                {"id": "e1", "description": "Event 1"},
            ]
        }
    ]

    mock_result = SimpleNamespace(id="e1", payload={"id": "e1"}, score=0.05)
    mock_vector_engine.search.return_value = [mock_result]

    with (
        patch.object(
            retriever, "extract_time_from_query", return_value=("2024-01-01", "2024-12-31")
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.temporal_retriever.CacheConfig") as mock_cache_config,
        patch("cognee.modules.retrieval.temporal_retriever.session_user") as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = None  # No user

        objects = await retriever.get_retrieved_objects("What happened in 2024?")
        context = await retriever.get_context_from_objects("What happened in 2024?", objects)
        completion = await retriever.get_completion_from_context(
            "What happened in 2024?", objects, context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1


@pytest.mark.asyncio
async def test_get_completion_with_response_model(mock_graph_engine, mock_vector_engine):
    """Test get_completion with custom response model."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        answer: str

    retriever = TemporalRetriever()

    mock_graph_engine.collect_time_ids.return_value = ["e1"]
    mock_graph_engine.collect_events.return_value = [
        {
            "events": [
                {"id": "e1", "description": "Event 1"},
            ]
        }
    ]

    mock_result = SimpleNamespace(id="e1", payload={"id": "e1"}, score=0.05)
    mock_vector_engine.search.return_value = [mock_result]

    with (
        patch.object(
            retriever, "extract_time_from_query", return_value=("2024-01-01", "2024-12-31")
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.generate_completion",
            return_value=TestModel(answer="Test answer"),
        ),
        patch("cognee.modules.retrieval.temporal_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects("What happened in 2024?")
        context = await retriever.get_context_from_objects("What happened in 2024?", objects)
        completion = await retriever.get_completion_from_context(
            "What happened in 2024?", objects, context
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert isinstance(completion[0], TestModel)


@pytest.mark.asyncio
async def test_extract_time_from_query_relative_path():
    """Test extract_time_from_query with relative prompt path."""
    retriever = TemporalRetriever(time_extraction_prompt_path="extract_query_time.txt")

    mock_timestamp_from = Timestamp(year=2024, month=1, day=1)
    mock_timestamp_to = Timestamp(year=2024, month=12, day=31)
    mock_interval = QueryInterval(starts_at=mock_timestamp_from, ends_at=mock_timestamp_to)

    with (
        patch("cognee.modules.retrieval.temporal_retriever.os.path.isabs", return_value=False),
        patch("cognee.modules.retrieval.temporal_retriever.datetime") as mock_datetime,
        patch(
            "cognee.modules.retrieval.temporal_retriever.render_prompt",
            return_value="System prompt",
        ),
        patch.object(
            LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            return_value=mock_interval,
        ),
    ):
        mock_datetime.now.return_value.strftime.return_value = "11-12-2024"

        time_from, time_to = await retriever.extract_time_from_query("What happened in 2024?")

    assert time_from == mock_timestamp_from
    assert time_to == mock_timestamp_to


@pytest.mark.asyncio
async def test_extract_time_from_query_absolute_path():
    """Test extract_time_from_query with absolute prompt path."""
    retriever = TemporalRetriever(
        time_extraction_prompt_path="/absolute/path/to/extract_query_time.txt"
    )

    mock_timestamp_from = Timestamp(year=2024, month=1, day=1)
    mock_timestamp_to = Timestamp(year=2024, month=12, day=31)
    mock_interval = QueryInterval(starts_at=mock_timestamp_from, ends_at=mock_timestamp_to)

    with (
        patch("cognee.modules.retrieval.temporal_retriever.os.path.isabs", return_value=True),
        patch(
            "cognee.modules.retrieval.temporal_retriever.os.path.dirname",
            return_value="/absolute/path/to",
        ),
        patch(
            "cognee.modules.retrieval.temporal_retriever.os.path.basename",
            return_value="extract_query_time.txt",
        ),
        patch("cognee.modules.retrieval.temporal_retriever.datetime") as mock_datetime,
        patch(
            "cognee.modules.retrieval.temporal_retriever.render_prompt",
            return_value="System prompt",
        ),
        patch.object(
            LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            return_value=mock_interval,
        ),
    ):
        mock_datetime.now.return_value.strftime.return_value = "11-12-2024"

        time_from, time_to = await retriever.extract_time_from_query("What happened in 2024?")

    assert time_from == mock_timestamp_from
    assert time_to == mock_timestamp_to


@pytest.mark.asyncio
async def test_extract_time_from_query_with_none_values():
    """Test extract_time_from_query when interval has None values."""
    retriever = TemporalRetriever(time_extraction_prompt_path="extract_query_time.txt")

    mock_interval = QueryInterval(starts_at=None, ends_at=None)

    with (
        patch("cognee.modules.retrieval.temporal_retriever.os.path.isabs", return_value=False),
        patch("cognee.modules.retrieval.temporal_retriever.datetime") as mock_datetime,
        patch(
            "cognee.modules.retrieval.temporal_retriever.render_prompt",
            return_value="System prompt",
        ),
        patch.object(
            LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            return_value=mock_interval,
        ),
    ):
        mock_datetime.now.return_value.strftime.return_value = "11-12-2024"

        time_from, time_to = await retriever.extract_time_from_query("What happened?")

    assert time_from is None
    assert time_to is None
