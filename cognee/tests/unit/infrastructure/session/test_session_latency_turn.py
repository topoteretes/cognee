from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.infrastructure.session.session_latency_turn import (
    commit_latency_turn,
    complete_latency_turn,
    load_latency_turn_snapshot,
)
from cognee.infrastructure.session.session_search_models import (
    SessionTurnSnapshot,
    get_session_search_completion_model,
)


@pytest.mark.asyncio
async def test_snapshot_loads_history_guidance_and_previous_served_context():
    manager = MagicMock()
    manager.is_auto_feedback_enabled.return_value = True
    manager.get_session = AsyncMock(
        return_value=[
            {
                "qa_id": "qa-1",
                "question": "first",
                "answer": "answer one",
            },
            {
                "qa_id": "qa-2",
                "question": "second",
                "answer": "answer two",
                "used_session_context_ids": ["ctx-1"],
            },
        ]
    )

    with (
        patch(
            "cognee.infrastructure.session.session_latency_turn.select_session_history",
            new_callable=AsyncMock,
            return_value="formatted history",
        ),
        patch(
            "cognee.infrastructure.session.session_latency_turn.build_active_context_block_safe",
            new_callable=AsyncMock,
            return_value=("active guidance", ["ctx-2"]),
        ),
        patch(
            "cognee.infrastructure.session.session_latency_turn.load_served_context_payload",
            new_callable=AsyncMock,
            return_value=[{"id": "ctx-1", "content": "previous guidance"}],
        ) as load_served,
    ):
        snapshot = await load_latency_turn_snapshot(
            manager,
            user_id="u1",
            session_id="s1",
            raw_message="current question",
        )

    assert snapshot.recent_qas == (
        ("qa-1", "first", "answer one"),
        ("qa-2", "second", "answer two"),
    )
    assert snapshot.completion_history == "formatted history"
    assert snapshot.active_context == "active guidance"
    assert snapshot.active_context_ids == ("ctx-2",)
    assert snapshot.previous_qa_id == "qa-2"
    assert snapshot.previous_served_context == (("ctx-1", "previous guidance"),)
    load_served.assert_awaited_once_with(
        manager,
        user_id="u1",
        session_id="s1",
        served_ids=["ctx-1"],
    )


@pytest.mark.asyncio
async def test_snapshot_skips_context_reads_when_auto_feedback_is_disabled():
    manager = MagicMock()
    manager.is_auto_feedback_enabled.return_value = False
    manager.get_session = AsyncMock(
        return_value=[
            {
                "qa_id": "qa-1",
                "question": "first",
                "answer": "answer",
                "used_session_context_ids": ["ctx-1"],
            }
        ]
    )

    with (
        patch(
            "cognee.infrastructure.session.session_latency_turn.select_session_history",
            new_callable=AsyncMock,
            return_value="formatted history",
        ),
        patch(
            "cognee.infrastructure.session.session_latency_turn.build_active_context_block_safe",
            new_callable=AsyncMock,
        ) as build_active,
        patch(
            "cognee.infrastructure.session.session_latency_turn.load_served_context_payload",
            new_callable=AsyncMock,
        ) as load_served,
    ):
        snapshot = await load_latency_turn_snapshot(
            manager,
            user_id="u1",
            session_id="s1",
            raw_message="current question",
        )

    assert snapshot.completion_history == "formatted history"
    assert snapshot.active_context == ""
    assert snapshot.previous_served_context == ()
    build_active.assert_not_awaited()
    load_served.assert_not_awaited()


@pytest.mark.asyncio
async def test_completion_appends_contract_and_uses_loaded_prompt_state():
    snapshot = SessionTurnSnapshot(
        raw_message="current question",
        completion_history="history",
        active_context="guidance",
    )
    completion_model = get_session_search_completion_model(str)

    with (
        patch(
            "cognee.infrastructure.session.session_latency_turn.read_query_prompt",
            return_value="contract",
        ),
        patch(
            "cognee.infrastructure.session.session_latency_turn.generate_completion",
            new_callable=AsyncMock,
            return_value=completion_model(
                response="answer",
                feedback_evidence=["correction"],
            ),
        ) as generate,
    ):
        completion = await complete_latency_turn(
            snapshot=snapshot,
            context="retrieved context",
            user_id="not-a-uuid",
            session_id="s1",
            user_prompt_path="user.txt",
            system_prompt_path="system.txt",
            system_prompt="caller system",
            response_model=str,
            auto_feedback=True,
        )

    assert completion.response == "answer"
    assert completion.feedback_evidence == ["correction"]
    call = generate.await_args.kwargs
    assert call["system_prompt"] == "caller system\n\ncontract"
    assert call["conversation_history"] == "guidance\n\nhistory"
    assert call["response_model"] is completion_model


@pytest.mark.asyncio
async def test_plain_completion_preserves_custom_response_without_evidence():
    class Answer(BaseModel):
        text: str

    answer = Answer(text="answer")
    with patch(
        "cognee.infrastructure.session.session_latency_turn.generate_completion",
        new_callable=AsyncMock,
        return_value=answer,
    ) as generate:
        completion = await complete_latency_turn(
            snapshot=SessionTurnSnapshot(raw_message="question", completion_history="history"),
            context="context",
            user_id="u1",
            session_id="s1",
            user_prompt_path="user.txt",
            system_prompt_path="system.txt",
            system_prompt=None,
            response_model=Answer,
            auto_feedback=False,
        )

    assert completion.response == answer
    assert completion.feedback_evidence == []
    assert generate.await_args.kwargs["response_model"] is Answer


@pytest.mark.asyncio
async def test_completion_tracks_only_uuid_users():
    class UsageScope:
        entered = False

        async def __aenter__(self):
            self.entered = True

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    scope = UsageScope()
    with (
        patch(
            "cognee.infrastructure.session.session_latency_turn.generate_completion",
            new_callable=AsyncMock,
            return_value="answer",
        ),
        patch(
            "cognee.infrastructure.session.session_latency_turn.track_session_usage",
            return_value=scope,
        ) as track,
    ):
        await complete_latency_turn(
            snapshot=SessionTurnSnapshot(raw_message="question"),
            context="context",
            user_id=(user_id := uuid4()),
            session_id="s1",
            user_prompt_path="user.txt",
            system_prompt_path="system.txt",
            system_prompt=None,
            response_model=str,
            auto_feedback=False,
        )

    track.assert_called_once_with("s1", user_id)
    assert scope.entered is True


@pytest.mark.asyncio
async def test_commit_stores_qa_before_evidence_and_returns_work_item():
    events = []
    manager = MagicMock()

    async def add_qa(**kwargs):
        events.append(("qa", kwargs))
        return "qa-current"

    async def create_evidence(**kwargs):
        events.append(("evidence", kwargs))
        return True

    manager.add_qa = AsyncMock(side_effect=add_qa)
    manager.create_session_context_entry = AsyncMock(side_effect=create_evidence)
    completion = get_session_search_completion_model(str)(
        response="answer",
        feedback_evidence=["correction"],
        future_context_evidence=["preference"],
    )
    snapshot = SessionTurnSnapshot(
        raw_message="question",
        active_context_ids=("ctx-1",),
        previous_qa_id="qa-previous",
        previous_question="previous",
        previous_answer="old answer",
        previous_served_context=(("ctx-old", "old guidance"),),
    )

    work_item = await commit_latency_turn(
        manager,
        snapshot=snapshot,
        completion=completion,
        user_id="u1",
        session_id="s1",
        dataset_id="d1",
        used_graph_element_ids={"node_ids": ["n1"]},
        auto_feedback=True,
    )

    assert [event[0] for event in events] == ["qa", "evidence"]
    assert events[0][1]["used_session_context_ids"] == ["ctx-1"]
    evidence = events[1][1]["entry_dump"]
    assert evidence["current_qa_id"] == "qa-current"
    assert evidence["previous_qa_id"] == "qa-previous"
    assert evidence["status"] == "pending"
    assert work_item.evidence_id == evidence["id"]


@pytest.mark.asyncio
async def test_acknowledgement_skips_qa_but_persists_evidence():
    manager = MagicMock()
    manager.add_qa = AsyncMock()
    manager.create_session_context_entry = AsyncMock(return_value=True)
    completion = get_session_search_completion_model(str)(
        response="Understood.",
        is_acknowledgement=True,
    )

    work_item = await commit_latency_turn(
        manager,
        snapshot=SessionTurnSnapshot(raw_message="Use concise answers."),
        completion=completion,
        user_id="u1",
        session_id="s1",
        dataset_id=None,
        used_graph_element_ids=None,
        auto_feedback=True,
    )

    manager.add_qa.assert_not_awaited()
    assert (
        manager.create_session_context_entry.await_args.kwargs["entry_dump"]["current_qa_id"]
        is None
    )
    assert work_item is not None


@pytest.mark.asyncio
async def test_evidence_storage_failure_returns_no_work_item_after_qa():
    manager = MagicMock()
    manager.add_qa = AsyncMock(return_value="qa-current")
    manager.create_session_context_entry = AsyncMock(return_value=False)

    work_item = await commit_latency_turn(
        manager,
        snapshot=SessionTurnSnapshot(raw_message="question"),
        completion=get_session_search_completion_model(str)(response="answer"),
        user_id="u1",
        session_id="s1",
        dataset_id=None,
        used_graph_element_ids=None,
        auto_feedback=True,
    )

    manager.add_qa.assert_awaited_once()
    assert work_item is None


@pytest.mark.asyncio
async def test_commit_without_auto_feedback_stores_only_qa():
    manager = MagicMock()
    manager.add_qa = AsyncMock(return_value="qa-current")
    manager.create_session_context_entry = AsyncMock()

    work_item = await commit_latency_turn(
        manager,
        snapshot=SessionTurnSnapshot(raw_message="question"),
        completion=get_session_search_completion_model(str)(response="answer"),
        user_id="u1",
        session_id="s1",
        dataset_id=None,
        used_graph_element_ids=None,
        auto_feedback=False,
    )

    manager.add_qa.assert_awaited_once()
    manager.create_session_context_entry.assert_not_awaited()
    assert work_item is None
