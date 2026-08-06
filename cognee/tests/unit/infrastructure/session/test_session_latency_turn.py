from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.session.session_latency_turn import load_latency_turn_snapshot


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
