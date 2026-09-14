from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.streaming.token_sink import (
    TokenSink,
    get_active_token_sink,
    requested_token_sink,
)
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_concurrent_turn import (
    SessionTurnContext,
    TurnPrompts,
    analyze_turn,
    commit_turn,
    complete_turn,
    load_turn_context,
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
            "cognee.infrastructure.session.session_concurrent_turn.select_session_history",
            new_callable=AsyncMock,
            return_value="formatted history",
        ),
        patch(
            "cognee.infrastructure.session.session_concurrent_turn.build_active_context_block_safe",
            new_callable=AsyncMock,
            return_value=("active guidance", ["ctx-2"]),
        ),
        patch(
            "cognee.infrastructure.session.session_concurrent_turn.load_served_context_payload",
            new_callable=AsyncMock,
            return_value=[{"id": "ctx-1", "content": "previous guidance"}],
        ) as load_served,
    ):
        snapshot = await load_turn_context(
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
            "cognee.infrastructure.session.session_concurrent_turn.select_session_history",
            new_callable=AsyncMock,
            return_value="formatted history",
        ),
        patch(
            "cognee.infrastructure.session.session_concurrent_turn.build_active_context_block_safe",
            new_callable=AsyncMock,
        ) as build_active,
        patch(
            "cognee.infrastructure.session.session_concurrent_turn.load_served_context_payload",
            new_callable=AsyncMock,
        ) as load_served,
    ):
        snapshot = await load_turn_context(
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
async def test_load_turn_context_fails_open_to_raw_message_only():
    manager = MagicMock()
    manager.is_auto_feedback_enabled.return_value = True
    manager.get_session = AsyncMock(side_effect=RuntimeError("cache down"))

    turn_context = await load_turn_context(
        manager,
        user_id="u1",
        session_id="s1",
        raw_message="question",
    )

    assert turn_context == SessionTurnContext(raw_message="question")


@pytest.mark.asyncio
async def test_analysis_reads_the_user_turn_and_the_context_it_may_rate():
    snapshot = SessionTurnContext(
        raw_message="that was wrong",
        previous_question="what is the limit?",
        previous_answer="ten",
        previous_served_context=(("ctx-1", "Be concise."),),
    )
    analysis = SessionTurnAnalysis(query_to_answer="restated")

    with patch(
        "cognee.infrastructure.session.session_concurrent_turn.analyze_turn_for_session_context",
        new_callable=AsyncMock,
        return_value=analysis,
    ) as analyze:
        assert await analyze_turn(snapshot) is analysis

    analyze.assert_awaited_once_with(
        "that was wrong",
        previous_question="what is the limit?",
        previous_answer="ten",
        served_context=[{"id": "ctx-1", "content": "Be concise."}],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("llm down"), TimeoutError()])
async def test_analysis_fails_open_to_no_context_updates(failure):
    with patch(
        "cognee.infrastructure.session.session_concurrent_turn.analyze_turn_for_session_context",
        new_callable=AsyncMock,
        side_effect=failure,
    ):
        analysis = await analyze_turn(SessionTurnContext(raw_message="question"))

    assert analysis.candidate_context_updates == []
    assert analysis.served_context_ratings == []


@pytest.mark.asyncio
async def test_answer_uses_the_callers_own_prompts_and_response_model():
    class Answer(BaseModel):
        text: str

    snapshot = SessionTurnContext(
        raw_message="question",
        active_context="active guidance",
        completion_history="history",
    )

    with patch(
        "cognee.infrastructure.session.session_concurrent_turn.generate_answer",
        new_callable=AsyncMock,
        return_value=Answer(text="structured"),
    ) as generate:
        answer = await complete_turn(
            snapshot=snapshot,
            context="context",
            user_id="not-a-uuid",
            session_id="s1",
            prompts=TurnPrompts(
                user_prompt_path="user.txt",
                system_prompt_path="system.txt",
                system_prompt="caller system prompt",
                response_model=Answer,
            ),
        )

    assert answer == Answer(text="structured")
    call = generate.await_args.kwargs
    # No wrapper model: the caller's own response contract, unchanged.
    assert call["response_model"] is Answer
    assert call["system_prompt"] == "caller system prompt"
    assert call["conversation_history"] == "active guidance\n\nhistory"


@pytest.mark.asyncio
async def test_answer_tracks_usage_only_for_uuid_users():
    snapshot = SessionTurnContext(raw_message="question")
    user_id = uuid4()

    with (
        patch(
            "cognee.infrastructure.session.session_concurrent_turn.generate_answer",
            new_callable=AsyncMock,
            return_value="answer",
        ),
        patch(
            "cognee.infrastructure.session.session_concurrent_turn.track_session_usage"
        ) as track_usage,
    ):
        await complete_turn(
            snapshot=snapshot,
            context="context",
            user_id=user_id,
            session_id="s1",
            prompts=TurnPrompts(
                user_prompt_path="user.txt",
                system_prompt_path="system.txt",
                system_prompt=None,
                response_model=str,
            ),
        )

    track_usage.assert_called_once_with("s1", user_id)


@pytest.mark.asyncio
async def test_commit_applies_the_analysis_then_stores_the_qa():
    order = []
    manager = MagicMock()
    manager.add_qa = AsyncMock(side_effect=lambda **kwargs: order.append("qa"))
    snapshot = SessionTurnContext(
        raw_message="question",
        active_context_ids=("ctx-served-now",),
        previous_qa_id="qa-1",
        previous_served_context=(("ctx-rated", "Be concise."),),
    )
    analysis = SessionTurnAnalysis(
        candidate_context_updates=[
            {"section": "rules", "content": "Cite sources.", "confidence": 0.9}
        ]
    )

    with patch(
        "cognee.infrastructure.session.session_concurrent_turn.apply_session_turn_analysis",
        new_callable=AsyncMock,
        side_effect=lambda *args, **kwargs: order.append("analysis"),
    ) as apply_analysis:
        await commit_turn(
            manager,
            snapshot=snapshot,
            analysis=analysis,
            answer="the answer",
            user_id="u1",
            session_id="s1",
            used_graph_element_ids={"node_ids": ["n1"]},
        )

    assert order == ["analysis", "qa"]
    # Ratings target the context served to the *previous* answer.
    assert apply_analysis.await_args.kwargs["served_ids"] == ["ctx-rated"]
    assert apply_analysis.await_args.kwargs["previous_qa_id"] == "qa-1"
    stored = manager.add_qa.await_args.kwargs
    assert stored["question"] == "question"
    assert stored["answer"] == "the answer"
    assert stored["context"] == ""
    # The QA records the context served to *this* answer.
    assert stored["used_session_context_ids"] == ["ctx-served-now"]
    assert stored["used_graph_element_ids"] == {"node_ids": ["n1"]}


@pytest.mark.asyncio
async def test_commit_fails_open_when_the_qa_write_raises():
    """The answer already exists by commit time, so a broken cache must not surface."""
    manager = MagicMock()
    manager.add_qa = AsyncMock(side_effect=RuntimeError("cache down"))

    with patch(
        "cognee.infrastructure.session.session_concurrent_turn.apply_session_turn_analysis",
        new_callable=AsyncMock,
    ):
        await commit_turn(
            manager,
            snapshot=SessionTurnContext(raw_message="question"),
            analysis=SessionTurnAnalysis(),
            answer="the answer",
            user_id="u1",
            session_id="s1",
            used_graph_element_ids=None,
        )

    manager.add_qa.assert_awaited_once()


@pytest.mark.asyncio
async def test_commit_serializes_a_custom_response_model_for_storage():
    class Answer(BaseModel):
        text: str

    manager = MagicMock()
    manager.add_qa = AsyncMock()

    with patch(
        "cognee.infrastructure.session.session_concurrent_turn.apply_session_turn_analysis",
        new_callable=AsyncMock,
    ):
        await commit_turn(
            manager,
            snapshot=SessionTurnContext(raw_message="question"),
            analysis=SessionTurnAnalysis(),
            answer=Answer(text="structured"),
            user_id="u1",
            session_id="s1",
            used_graph_element_ids=None,
        )

    assert manager.add_qa.await_args.kwargs["answer"] == '{"text":"structured"}'


@pytest.mark.asyncio
async def test_the_turn_answer_is_the_call_a_listening_client_watches():
    """complete_turn must reach the LLM through generate_answer.

    The whole streaming feature rests on this one call being the answer call.
    Point it back at generate_completion and SSE goes silent while every other
    test in this file still passes — the turn is still produced, still tracked,
    still committed. Nothing else asserts it, so this is the guard.

    It deliberately does not mock generate_answer: the assertion is that the
    promotion is live *around the real completion*, which a mock would hide.
    """
    observed = {}

    async def _completion_that_looks_around(**_kwargs):
        observed["active"] = get_active_token_sink()
        return "answer"

    sink = TokenSink()
    token = requested_token_sink.set(sink)
    try:
        with (
            patch(
                "cognee.infrastructure.llm.config.get_llm_context_config",
                return_value=SimpleNamespace(llm_answer_streaming=True),
            ),
            patch(
                "cognee.modules.retrieval.utils.completion.generate_completion",
                new=AsyncMock(side_effect=_completion_that_looks_around),
            ),
            # Pin the adapter capability: unpinned it reads the ambient LLM
            # config, so whether this test promotes at all would depend on
            # whatever provider the host happens to have configured.
            patch.object(LLMGateway, "supports_answer_streaming", return_value=True),
        ):
            answer = await complete_turn(
                snapshot=SessionTurnContext(raw_message="question"),
                context="context",
                user_id="not-a-uuid",
                session_id="s1",
                prompts=TurnPrompts(
                    user_prompt_path="user.txt",
                    system_prompt_path="system.txt",
                    system_prompt=None,
                    response_model=str,
                ),
            )
    finally:
        requested_token_sink.reset(token)

    assert answer == "answer"
    assert observed["active"] is sink, (
        "the turn's answer did not run inside answer_scope — complete_turn is no "
        "longer routed through generate_answer, so nothing will stream"
    )


@pytest.mark.asyncio
async def test_a_turn_does_not_stream_when_nobody_asked():
    """The same path with no sink installed promotes nothing.

    Streaming is never load-bearing: with no listener the turn runs exactly as
    it did before the feature existed.
    """
    observed = {}

    async def _completion_that_looks_around(**_kwargs):
        observed["active"] = get_active_token_sink()
        return "answer"

    with (
        patch(
            "cognee.infrastructure.llm.config.get_llm_context_config",
            return_value=SimpleNamespace(llm_answer_streaming=True),
        ),
        patch(
            "cognee.modules.retrieval.utils.completion.generate_completion",
            new=AsyncMock(side_effect=_completion_that_looks_around),
        ),
        # Pin the adapter capability: unpinned it reads the ambient LLM
        # config, so whether this test promotes at all would depend on
        # whatever provider the host happens to have configured.
        patch.object(LLMGateway, "supports_answer_streaming", return_value=True),
    ):
        answer = await complete_turn(
            snapshot=SessionTurnContext(raw_message="question"),
            context="context",
            user_id="not-a-uuid",
            session_id="s1",
            prompts=TurnPrompts(
                user_prompt_path="user.txt",
                system_prompt_path="system.txt",
                system_prompt=None,
                response_model=str,
            ),
        )

    assert answer == "answer"
    assert observed["active"] is None
