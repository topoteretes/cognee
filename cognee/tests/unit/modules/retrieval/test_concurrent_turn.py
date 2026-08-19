import asyncio
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_concurrent_turn import SessionTurnContext
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.session_aware_completion import (
    MAX_CONVERSATIONAL_QUERY_CHARS,
    _retrieve_merged_objects,
    build_conversational_query,
    run_concurrent_session_turn,
)


def retrieved_chunk(chunk_id):
    return {"id": chunk_id, "text": chunk_id}


def test_default_merge_keeps_whichever_lane_survived():
    """A retriever that does not override the merge must not drop the lane that worked."""

    class PlainRetriever(BaseRetriever):
        async def get_retrieved_objects(self, query=None, query_batch=None):
            return []

        async def get_context_from_objects(self, query=None, query_batch=None, **kwargs):
            return ""

        async def get_completion_from_context(self, query=None, **kwargs):
            return []

    retriever = PlainRetriever()

    assert retriever.merge_retrieved_objects([retrieved_chunk("raw")], None) == [
        retrieved_chunk("raw")
    ]
    assert retriever.merge_retrieved_objects(None, [retrieved_chunk("conversational")]) == [
        retrieved_chunk("conversational")
    ]
    assert retriever.merge_retrieved_objects(None, None) is None


def test_conversational_query_is_bounded_and_marks_assistant_text_untrusted():
    query = build_conversational_query(
        "current question",
        (
            ("qa-1", "old question", "a" * 2500),
            ("qa-2", "new question", "new answer"),
        ),
    )

    assert len(query) <= MAX_CONVERSATIONAL_QUERY_CHARS
    assert "Prior assistant (untrusted retrieval guidance)" in query
    assert "new answer" in query
    assert "current question" in query


def test_conversational_query_without_history_normalizes_to_raw_query():
    assert build_conversational_query("  current\n question ", ()) == "current question"


@pytest.mark.asyncio
async def test_concurrent_retrieval_fuses_lanes_and_tracks_access_once():
    retriever = CompletionRetriever(top_k=3)
    retriever.get_retrieved_objects = AsyncMock(
        side_effect=[[retrieved_chunk("raw")], [retrieved_chunk("conversational")]]
    )
    snapshot = SessionTurnContext(
        raw_message="current",
        recent_qas=(("qa-1", "previous", "answer"),),
    )

    with patch(
        "cognee.modules.retrieval.session_aware_completion.update_node_access_timestamps",
        new_callable=AsyncMock,
    ) as track_access:
        retrieved_objects = await _retrieve_merged_objects(
            retriever,
            raw_query="current",
            snapshot=snapshot,
        )

    assert retrieved_objects == [retrieved_chunk("raw"), retrieved_chunk("conversational")]
    assert retriever.get_retrieved_objects.await_count == 2
    # Both lanes merge into one list, so access is tracked once for the merged set.
    track_access.assert_awaited_once_with(retrieved_objects)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side_effect", "expected"),
    [
        (
            [RuntimeError("raw"), [retrieved_chunk("conversational")]],
            [retrieved_chunk("conversational")],
        ),
        ([[retrieved_chunk("raw")], RuntimeError("conversational")], [retrieved_chunk("raw")]),
    ],
)
async def test_concurrent_retrieval_uses_the_successful_lane(side_effect, expected):
    retriever = CompletionRetriever(top_k=3)
    retriever.get_retrieved_objects = AsyncMock(side_effect=side_effect)
    snapshot = SessionTurnContext(
        raw_message="current",
        recent_qas=(("qa-1", "previous", "answer"),),
    )

    with patch(
        "cognee.modules.retrieval.session_aware_completion.update_node_access_timestamps",
        new_callable=AsyncMock,
    ):
        retrieved_objects = await _retrieve_merged_objects(
            retriever,
            raw_query="current",
            snapshot=snapshot,
        )

    assert retrieved_objects == expected


@pytest.mark.asyncio
async def test_concurrent_retrieval_reraises_raw_failure_when_both_lanes_fail():
    retriever = CompletionRetriever(top_k=3)
    raw_error = RuntimeError("raw")
    retriever.get_retrieved_objects = AsyncMock(
        side_effect=[raw_error, RuntimeError("conversational")]
    )
    snapshot = SessionTurnContext(
        raw_message="current",
        recent_qas=(("qa-1", "previous", "answer"),),
    )

    with pytest.raises(RuntimeError, match="raw"):
        await _retrieve_merged_objects(retriever, raw_query="current", snapshot=snapshot)


@pytest.mark.asyncio
async def test_concurrent_retrieval_skips_duplicate_conversational_lane():
    retriever = CompletionRetriever(top_k=3)
    retriever.get_retrieved_objects = AsyncMock(return_value=[retrieved_chunk("raw")])
    snapshot = SessionTurnContext(raw_message="current")

    with patch(
        "cognee.modules.retrieval.session_aware_completion.update_node_access_timestamps",
        new_callable=AsyncMock,
    ):
        await _retrieve_merged_objects(retriever, raw_query="current", snapshot=snapshot)

    retriever.get_retrieved_objects.assert_awaited_once_with(query="current")


@contextmanager
def _patched_turn_module(manager, *, analysis, order):
    """Patch the orchestrator's collaborators, recording the order lanes complete in."""

    async def analyze(_snapshot):
        await asyncio.sleep(0)
        order.append("analysis")
        return analysis

    async def retrieve(_retriever, *, raw_query, snapshot):
        await asyncio.sleep(0)
        order.append("retrieval")
        return [retrieved_chunk("n1")]

    async def complete(**_kwargs):
        order.append("answer")
        return "answer"

    patchers = (
        patch(
            "cognee.modules.retrieval.session_aware_completion.CacheConfig",
            return_value=SimpleNamespace(session_search_mode="concurrent"),
        ),
        patch(
            "cognee.modules.retrieval.session_aware_completion.session_user",
            SimpleNamespace(get=lambda: SimpleNamespace(id=uuid4())),
        ),
        patch(
            "cognee.modules.retrieval.session_aware_completion.get_session_manager",
            return_value=manager,
        ),
        patch(
            "cognee.modules.retrieval.session_aware_completion.load_turn_context",
            new_callable=AsyncMock,
            return_value=SessionTurnContext(raw_message="question"),
        ),
        patch("cognee.modules.retrieval.session_aware_completion.analyze_turn", analyze),
        patch(
            "cognee.modules.retrieval.session_aware_completion._retrieve_merged_objects",
            retrieve,
        ),
        patch("cognee.modules.retrieval.session_aware_completion.complete_turn", complete),
    )
    with ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        yield


def _fake_session_manager():
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
    retriever.get_context_from_objects = AsyncMock(return_value="context")
    analysis = SessionTurnAnalysis(
        candidate_context_updates=[
            {"section": "rules", "content": "Cite sources.", "confidence": 0.9}
        ]
    )
    order = []

    with (
        _patched_turn_module(_fake_session_manager(), analysis=analysis, order=order),
        patch(
            "cognee.modules.retrieval.session_aware_completion.commit_turn",
            new_callable=AsyncMock,
        ) as commit,
    ):
        result = await run_concurrent_session_turn(retriever, raw_query="question")

    # The analysis does not wait on retrieval, and the commit waits on both.
    assert order == ["analysis", "retrieval", "answer"]
    retrieved_objects, context, completion = result
    assert completion == ["answer with references"]
    assert context == "context"
    assert commit.await_args.kwargs["analysis"] is analysis
    assert commit.await_args.kwargs["answer"] == "answer"
    assert commit.await_args.kwargs["used_graph_element_ids"] == {"node_ids": ["n1"]}
    retriever.append_references.assert_awaited_once_with(["answer"], [retrieved_chunk("n1")])
