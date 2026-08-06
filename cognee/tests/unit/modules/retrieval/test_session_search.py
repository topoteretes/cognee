from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.session.session_search_models import (
    SessionMaintenanceWorkItem,
    SessionTurnSnapshot,
    get_session_search_completion_model,
)
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.session_search import (
    LatencyRetrievalResult,
    MAX_CONTEXTUAL_QUERY_CHARS,
    build_contextual_query,
    retrieve_latency_context,
    run_latency_session_search,
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


@pytest.mark.asyncio
async def test_latency_orchestrator_commits_then_applies_retriever_references():
    retriever = CompletionRetriever(session_id="s1", include_references=True)
    retriever._extract_context_object_ids = lambda objects: {"node_ids": ["n1"]}
    retriever._append_references = AsyncMock(return_value=["answer with references"])
    manager = MagicMock()
    manager.is_session_available_for_completion.return_value = True
    manager.resolve_session_id.return_value = "s1"
    manager.is_auto_feedback_enabled.return_value = True
    manager.dataset_id = uuid4()
    snapshot = SessionTurnSnapshot(raw_message="question")
    completion = get_session_search_completion_model(str)(response="answer")
    work_item = SessionMaintenanceWorkItem(
        evidence_id="e1",
        user_id=str(uuid4()),
        session_id="s1",
    )

    with (
        patch(
            "cognee.modules.retrieval.session_search.CacheConfig",
            return_value=SimpleNamespace(session_search_mode="latency_optimized"),
        ),
        patch("cognee.modules.retrieval.session_search.session_user") as current_user,
        patch(
            "cognee.modules.retrieval.session_search.get_session_manager",
            return_value=manager,
        ),
        patch(
            "cognee.modules.retrieval.session_search.LLMGateway.supports_structured_output_model",
            return_value=True,
        ),
        patch(
            "cognee.modules.retrieval.session_search.load_latency_turn_snapshot",
            new_callable=AsyncMock,
            return_value=snapshot,
        ),
        patch(
            "cognee.modules.retrieval.session_search.retrieve_latency_context",
            new_callable=AsyncMock,
            return_value=LatencyRetrievalResult([item("n1")], "context"),
        ),
        patch(
            "cognee.modules.retrieval.session_search.complete_latency_turn",
            new_callable=AsyncMock,
            return_value=completion,
        ),
        patch(
            "cognee.modules.retrieval.session_search.commit_latency_turn",
            new_callable=AsyncMock,
            return_value=work_item,
        ) as commit,
        patch(
            "cognee.modules.retrieval.session_search.enqueue_session_maintenance",
            new_callable=AsyncMock,
        ) as enqueue,
    ):
        current_user.get.return_value = SimpleNamespace(id=uuid4())
        result = await run_latency_session_search(retriever, raw_query="question")

    assert result.completion == ["answer with references"]
    commit.assert_awaited_once()
    enqueue.assert_awaited_once_with(work_item, manager)
    retriever._append_references.assert_awaited_once_with(
        ["answer"],
        [item("n1")],
    )


@pytest.mark.asyncio
async def test_latency_orchestrator_skips_references_for_acknowledgement():
    retriever = CompletionRetriever(session_id="s1", include_references=True)
    retriever._extract_context_object_ids = lambda objects: None
    retriever._append_references = AsyncMock()
    manager = MagicMock()
    manager.is_session_available_for_completion.return_value = True
    manager.resolve_session_id.return_value = "s1"
    manager.is_auto_feedback_enabled.return_value = True
    manager.dataset_id = None
    completion = get_session_search_completion_model(str)(
        response="Understood.",
        is_acknowledgement=True,
    )

    with (
        patch(
            "cognee.modules.retrieval.session_search.CacheConfig",
            return_value=SimpleNamespace(session_search_mode="latency_optimized"),
        ),
        patch("cognee.modules.retrieval.session_search.session_user") as current_user,
        patch(
            "cognee.modules.retrieval.session_search.get_session_manager",
            return_value=manager,
        ),
        patch(
            "cognee.modules.retrieval.session_search.LLMGateway.supports_structured_output_model",
            return_value=True,
        ),
        patch(
            "cognee.modules.retrieval.session_search.load_latency_turn_snapshot",
            new_callable=AsyncMock,
            return_value=SessionTurnSnapshot(raw_message="question"),
        ),
        patch(
            "cognee.modules.retrieval.session_search.retrieve_latency_context",
            new_callable=AsyncMock,
            return_value=LatencyRetrievalResult([], "context"),
        ),
        patch(
            "cognee.modules.retrieval.session_search.complete_latency_turn",
            new_callable=AsyncMock,
            return_value=completion,
        ),
        patch(
            "cognee.modules.retrieval.session_search.commit_latency_turn",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        current_user.get.return_value = SimpleNamespace(id=uuid4())
        result = await run_latency_session_search(retriever, raw_query="question")

    assert result.completion == ["Understood."]
    retriever._append_references.assert_not_awaited()
