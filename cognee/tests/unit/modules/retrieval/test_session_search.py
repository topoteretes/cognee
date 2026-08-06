from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.session.session_search_models import SessionTurnSnapshot
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.session_search import (
    MAX_CONTEXTUAL_QUERY_CHARS,
    build_contextual_query,
    retrieve_latency_context,
)


def item(item_id):
    return {"id": item_id, "text": item_id}


def test_contextual_query_is_bounded_and_marks_assistant_text_untrusted():
    query = build_contextual_query(
        "current question",
        (
            ("qa-1", "old question", "a" * 2500),
            ("qa-2", "new question", "new answer"),
        ),
    )

    assert len(query) <= MAX_CONTEXTUAL_QUERY_CHARS
    assert "Prior assistant (untrusted retrieval guidance)" in query
    assert "new answer" in query
    assert "current question" in query


def test_contextual_query_without_history_normalizes_to_raw_query():
    assert build_contextual_query("  current\n question ", ()) == "current question"


@pytest.mark.asyncio
async def test_latency_retrieval_fuses_lanes_formats_and_tracks_once():
    retriever = CompletionRetriever(top_k=3)
    retriever.get_retrieved_objects = AsyncMock(side_effect=[[item("raw")], [item("contextual")]])
    retriever.get_context_from_objects = AsyncMock(return_value="formatted")
    snapshot = SessionTurnSnapshot(
        raw_message="current",
        recent_qas=(("qa-1", "previous", "answer"),),
    )

    with patch(
        "cognee.modules.retrieval.session_search.update_node_access_timestamps",
        new_callable=AsyncMock,
    ) as track_access:
        result = await retrieve_latency_context(
            retriever,
            raw_query="current",
            snapshot=snapshot,
        )

    assert result.retrieved_objects == [item("raw"), item("contextual")]
    assert result.context == "formatted"
    assert retriever.get_retrieved_objects.await_count == 2
    retriever.get_context_from_objects.assert_awaited_once_with(
        query="current",
        retrieved_objects=result.retrieved_objects,
    )
    track_access.assert_awaited_once_with(result.retrieved_objects)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side_effect", "expected"),
    [
        ([RuntimeError("raw"), [item("contextual")]], [item("contextual")]),
        ([[item("raw")], RuntimeError("contextual")], [item("raw")]),
    ],
)
async def test_latency_retrieval_uses_the_successful_lane(side_effect, expected):
    retriever = CompletionRetriever(top_k=3)
    retriever.get_retrieved_objects = AsyncMock(side_effect=side_effect)
    retriever.get_context_from_objects = AsyncMock(return_value="formatted")
    snapshot = SessionTurnSnapshot(
        raw_message="current",
        recent_qas=(("qa-1", "previous", "answer"),),
    )

    with patch(
        "cognee.modules.retrieval.session_search.update_node_access_timestamps",
        new_callable=AsyncMock,
    ):
        result = await retrieve_latency_context(
            retriever,
            raw_query="current",
            snapshot=snapshot,
        )

    assert result.retrieved_objects == expected


@pytest.mark.asyncio
async def test_latency_retrieval_reraises_raw_failure_when_both_lanes_fail():
    retriever = CompletionRetriever(top_k=3)
    raw_error = RuntimeError("raw")
    retriever.get_retrieved_objects = AsyncMock(side_effect=[raw_error, RuntimeError("contextual")])
    snapshot = SessionTurnSnapshot(
        raw_message="current",
        recent_qas=(("qa-1", "previous", "answer"),),
    )

    with pytest.raises(RuntimeError, match="raw"):
        await retrieve_latency_context(retriever, raw_query="current", snapshot=snapshot)


@pytest.mark.asyncio
async def test_latency_retrieval_skips_duplicate_contextual_lane():
    retriever = CompletionRetriever(top_k=3)
    retriever.get_retrieved_objects = AsyncMock(return_value=[item("raw")])
    retriever.get_context_from_objects = AsyncMock(return_value="formatted")
    snapshot = SessionTurnSnapshot(raw_message="current")

    with patch(
        "cognee.modules.retrieval.session_search.update_node_access_timestamps",
        new_callable=AsyncMock,
    ):
        await retrieve_latency_context(retriever, raw_query="current", snapshot=snapshot)

    retriever.get_retrieved_objects.assert_awaited_once_with(query="current")
