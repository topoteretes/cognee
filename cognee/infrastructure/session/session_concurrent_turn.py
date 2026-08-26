"""Session reads and writes for one concurrent turn.

A concurrent turn does not chain analysis before retrieval the way the sequential path does.
It reads the session once, then runs the turn analysis and the answer concurrently, and
applies the analysis after both land. These are the session-side pieces of that: the
turn context both lanes read from, the analysis lane, the answer call, and the commit.
Public coroutines are fail-open so they never block answer generation.
"""

import asyncio
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from cognee.infrastructure.session.feedback_detection import analyze_turn_for_session_context
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_context_builder import render_preference_block
from cognee.infrastructure.session.session_turn import (
    apply_session_turn_analysis,
    build_active_context_block_safe,
    coerce_qa_entry,
    compose_session_prompt,
    load_preference_lines_safe,
    load_served_context_payload,
    select_session_history,
)
from cognee.modules.retrieval.utils.completion import generate_answer
from cognee.modules.session_lifecycle import track_session_usage
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_concurrent_turn")

# The analysis runs alongside the answer, so it normally finishes first. This only bounds
# the pathological case where it would hold the turn open past its own answer.
ANALYSIS_TIMEOUT_SECONDS = 30.0


class SessionTurnContext(BaseModel):
    """Immutable session read used by both lanes of a concurrent turn.

    A concurrent turn analyzes the user's message and answers it at the same time, so both
    lanes have to work from the same reading of the session. Loading it once also keeps
    the two lanes from racing each other's cache reads.
    """

    model_config = ConfigDict(frozen=True)

    raw_message: str
    # (qa_id, question, answer) for the last two turns; builds the conversational query.
    recent_qas: tuple[tuple[str, str, str], ...] = ()
    completion_history: str = ""
    active_context: str = ""
    active_context_ids: tuple[str, ...] = ()
    previous_qa_id: str | None = None
    previous_question: str | None = None
    previous_answer: str | None = None
    # (entry_id, content) served to the previous answer; the analysis lane rates these.
    previous_served_context: tuple[tuple[str, str], ...] = ()


class TurnPrompts(BaseModel):
    """The answer settings a retriever owns, passed by value.

    Concurrent turns answer with the caller's own prompts, but this layer must not import
    retriever types, so the four fields travel on their own.
    """

    model_config = ConfigDict(frozen=True)

    user_prompt_path: str
    system_prompt_path: str
    system_prompt: str | None = None
    response_model: type = str


async def load_turn_context(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    raw_message: str,
) -> SessionTurnContext:
    """Read every piece of session state one concurrent turn needs, in one pass.

    Fail open to a raw-message-only context on any cache error so the turn can still
    answer without session extras (same contract as sequential prepare).
    """
    try:
        auto_feedback = session_manager.is_auto_feedback_enabled()
        preference_lines = await load_preference_lines_safe()
        loads = [
            session_manager.get_session(
                user_id=user_id,
                session_id=session_id,
                formatted=False,
                last_n=2,
            ),
            select_session_history(
                session_manager,
                user_id=user_id,
                session_id=session_id,
                query_text=raw_message,
            ),
        ]
        if auto_feedback:
            loads.append(
                build_active_context_block_safe(
                    session_manager,
                    user_id=user_id,
                    session_id=session_id,
                    query=raw_message,
                    preference_lines=preference_lines,
                )
            )
        # The guidance block is the only optional load, so it is the only trailing result.
        recent_entries, completion_history, *optional = await asyncio.gather(*loads)
        if optional:
            active_context, active_context_ids = optional[0]
        else:
            # No stored-entry guidance layer; durable preferences still render
            # through the same owner, budgets, and block shape.
            active_context = render_preference_block(preference_lines) if preference_lines else ""
            active_context_ids = []

        # get_session already caps at last_n=2 above; recent_qas inherits that bound.
        recent_rows = [coerce_qa_entry(entry) for entry in recent_entries or []]
        recent_qas = tuple(
            (
                str(row.get("qa_id") or ""),
                str(row.get("question") or ""),
                str(row.get("answer") or ""),
            )
            for row in recent_rows
        )
        previous = recent_rows[-1] if recent_rows else {}
        previous_served_ids = previous.get("used_session_context_ids") or []
        if not isinstance(previous_served_ids, list):
            previous_served_ids = []

        previous_served_context = []
        if auto_feedback and previous_served_ids:
            previous_served_context = await load_served_context_payload(
                session_manager,
                user_id=user_id,
                session_id=session_id,
                served_ids=[str(entry_id) for entry_id in previous_served_ids],
            )

        return SessionTurnContext(
            raw_message=raw_message,
            recent_qas=recent_qas,
            completion_history=completion_history if isinstance(completion_history, str) else "",
            active_context=active_context,
            active_context_ids=tuple(str(entry_id) for entry_id in active_context_ids),
            previous_qa_id=str(previous.get("qa_id")) if previous.get("qa_id") else None,
            previous_question=previous.get("question"),
            previous_answer=previous.get("answer"),
            previous_served_context=tuple(
                (str(entry.get("id")), str(entry.get("content") or ""))
                for entry in previous_served_context
                if isinstance(entry, dict) and entry.get("id") is not None
            ),
        )
    except Exception as error:
        logger.warning("Concurrent turn context load failed open: %s", error)
        return SessionTurnContext(raw_message=raw_message)


async def analyze_turn(snapshot: SessionTurnContext) -> SessionTurnAnalysis:
    """Analyze one turn under a timeout. Fail open to no context updates.

    Concurrent mode uses only the two context-maintenance outputs; the routing fields are
    ignored because retrieval and the answer are already in flight by the time this lands.
    """
    try:
        return await asyncio.wait_for(
            analyze_turn_for_session_context(
                snapshot.raw_message,
                previous_question=snapshot.previous_question,
                previous_answer=snapshot.previous_answer,
                served_context=[
                    {"id": entry_id, "content": content}
                    for entry_id, content in snapshot.previous_served_context
                ],
            ),
            timeout=ANALYSIS_TIMEOUT_SECONDS,
        )
    except Exception as error:
        logger.warning("Concurrent turn analysis failed open: %s", error)
        return SessionTurnAnalysis()


async def complete_turn(
    *,
    snapshot: SessionTurnContext,
    context: Any,
    user_id: Any,
    session_id: str,
    prompts: TurnPrompts,
) -> Any:
    """Generate the turn's answer from retrieval context and session prompt history."""
    completion_call = generate_answer(
        query=snapshot.raw_message,
        context=context,
        user_prompt_path=prompts.user_prompt_path,
        system_prompt_path=prompts.system_prompt_path,
        system_prompt=prompts.system_prompt,
        conversation_history=compose_session_prompt(
            snapshot.active_context,
            snapshot.completion_history,
        ),
        response_model=prompts.response_model,
    )

    # generate_answer rather than generate_completion: this call *is* the turn's
    # answer, so it is the one a listening client may watch. That is the entire
    # distinction, and it lives in the name — this module neither imports nor
    # needs to know about streaming. Turn analysis runs concurrently via
    # asyncio.gather, which snapshots the context per task, so the analysis lane
    # cannot inherit the answer lane's stream.
    if isinstance(user_id, UUID):
        async with track_session_usage(session_id, user_id):
            return await completion_call
    return await completion_call


async def commit_turn(
    session_manager,
    *,
    snapshot: SessionTurnContext,
    analysis: SessionTurnAnalysis,
    answer: Any,
    user_id: str,
    session_id: str,
    used_graph_element_ids: dict | None,
) -> None:
    """Apply the turn's context updates, then store the QA pair.

    Both fail open. The answer is already generated by the time this runs, so a failing
    cache write costs the recording, never the answer. ``apply_session_turn_analysis``
    swallows its own errors; ``add_qa`` does not, so it is guarded here.
    """
    await apply_session_turn_analysis(
        session_manager,
        user_id=user_id,
        session_id=session_id,
        query=snapshot.raw_message,
        analysis=analysis,
        previous_qa_id=snapshot.previous_qa_id,
        served_ids=[entry_id for entry_id, _content in snapshot.previous_served_context],
    )
    try:
        await session_manager.add_qa(
            user_id=user_id,
            question=snapshot.raw_message,
            context="",
            answer=answer.model_dump_json() if isinstance(answer, BaseModel) else str(answer),
            session_id=session_id,
            used_graph_element_ids=used_graph_element_ids,
            used_session_context_ids=list(snapshot.active_context_ids) or None,
        )
    except Exception as error:
        logger.warning("Concurrent turn QA write failed open: %s", error)
