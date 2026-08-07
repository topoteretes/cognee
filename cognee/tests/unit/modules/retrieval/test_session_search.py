import asyncio
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.session.session_search_models import SessionTurnSnapshot
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.modules.retrieval.session_search import (
    MAX_CONTEXTUAL_QUERY_CHARS,
    build_contextual_query,
    retrieve_turn_context,
    try_concurrent_turn,
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
async def test_concurrent_retrieval_fuses_lanes_formats_and_tracks_once():
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
        retrieved_objects, context = await retrieve_turn_context(
            retriever,
            raw_query="current",
            snapshot=snapshot,
        )

    assert retrieved_objects == [item("raw"), item("contextual")]
    assert context == "formatted"
    assert retriever.get_retrieved_objects.await_count == 2
    retriever.get_context_from_objects.assert_awaited_once_with(
        query="current",
        retrieved_objects=retrieved_objects,
    )
    track_access.assert_awaited_once_with(retrieved_objects)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side_effect", "expected"),
    [
        ([RuntimeError("raw"), [item("contextual")]], [item("contextual")]),
        ([[item("raw")], RuntimeError("contextual")], [item("raw")]),
    ],
)
async def test_concurrent_retrieval_uses_the_successful_lane(side_effect, expected):
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
        retrieved_objects, _context = await retrieve_turn_context(
            retriever,
            raw_query="current",
            snapshot=snapshot,
        )

    assert retrieved_objects == expected


@pytest.mark.asyncio
async def test_concurrent_retrieval_reraises_raw_failure_when_both_lanes_fail():
    retriever = CompletionRetriever(top_k=3)
    raw_error = RuntimeError("raw")
    retriever.get_retrieved_objects = AsyncMock(side_effect=[raw_error, RuntimeError("contextual")])
    snapshot = SessionTurnSnapshot(
        raw_message="current",
        recent_qas=(("qa-1", "previous", "answer"),),
    )

    with pytest.raises(RuntimeError, match="raw"):
        await retrieve_turn_context(retriever, raw_query="current", snapshot=snapshot)


@pytest.mark.asyncio
async def test_concurrent_retrieval_skips_duplicate_contextual_lane():
    retriever = CompletionRetriever(top_k=3)
    retriever.get_retrieved_objects = AsyncMock(return_value=[item("raw")])
    retriever.get_context_from_objects = AsyncMock(return_value="formatted")
    snapshot = SessionTurnSnapshot(raw_message="current")

    with patch(
        "cognee.modules.retrieval.session_search.update_node_access_timestamps",
        new_callable=AsyncMock,
    ):
        await retrieve_turn_context(retriever, raw_query="current", snapshot=snapshot)

    retriever.get_retrieved_objects.assert_awaited_once_with(query="current")


@contextmanager
def _concurrent_environment(manager, *, analysis, order):
    """Patch the orchestrator's collaborators, recording the order lanes complete in."""

    async def analyze(_snapshot):
        await asyncio.sleep(0)
        order.append("analysis")
        return analysis

    async def retrieve(_retriever, *, raw_query, snapshot):
        await asyncio.sleep(0)
        order.append("retrieval")
        return [item("n1")], "context"

    async def complete(**_kwargs):
        order.append("answer")
        return "answer"

    patchers = (
        patch(
            "cognee.modules.retrieval.session_search.CacheConfig",
            return_value=SimpleNamespace(session_search_mode="concurrent"),
        ),
        patch(
            "cognee.modules.retrieval.session_search.session_user",
            SimpleNamespace(get=lambda: SimpleNamespace(id=uuid4())),
        ),
        patch(
            "cognee.modules.retrieval.session_search.get_session_manager",
            return_value=manager,
        ),
        patch(
            "cognee.modules.retrieval.session_search.load_turn_snapshot",
            new_callable=AsyncMock,
            return_value=SessionTurnSnapshot(raw_message="question"),
        ),
        patch("cognee.modules.retrieval.session_search.analyze_turn_concurrently", analyze),
        patch("cognee.modules.retrieval.session_search.retrieve_turn_context", retrieve),
        patch("cognee.modules.retrieval.session_search.complete_turn", complete),
    )
    with ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        yield


def _session_manager():
    manager = MagicMock()
    manager.is_session_available_for_completion.return_value = True
    manager.resolve_session_id.return_value = "s1"
    manager.is_auto_feedback_enabled.return_value = True
    return manager


@pytest.mark.asyncio
async def test_analysis_runs_alongside_retrieval_and_commits_after_both():
    retriever = CompletionRetriever(session_id="s1", include_references=True)
    retriever.extract_context_object_ids = lambda objects: {"node_ids": ["n1"]}
    retriever.append_references = AsyncMock(return_value=["answer with references"])
    analysis = SessionTurnAnalysis(
        candidate_context_updates=[
            {"section": "rules", "content": "Cite sources.", "confidence": 0.9}
        ]
    )
    order = []

    with (
        _concurrent_environment(_session_manager(), analysis=analysis, order=order),
        patch(
            "cognee.modules.retrieval.session_search.commit_turn",
            new_callable=AsyncMock,
        ) as commit,
    ):
        result = await try_concurrent_turn(retriever, raw_query="question")

    # The analysis does not wait on retrieval, and the commit waits on both.
    assert order == ["analysis", "retrieval", "answer"]
    assert result.completion == ["answer with references"]
    assert commit.await_args.kwargs["analysis"] is analysis
    assert commit.await_args.kwargs["answer"] == "answer"
    assert commit.await_args.kwargs["used_graph_element_ids"] == {"node_ids": ["n1"]}
    retriever.append_references.assert_awaited_once_with(["answer"], [item("n1")])
