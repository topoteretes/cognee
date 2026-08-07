"""Session reads and writes for one latency-optimized turn.

A latency turn does not chain analysis before retrieval the way the accuracy path does.
It reads the session once, then runs the turn analysis and the answer concurrently, and
applies the analysis after both land. These are the session-side pieces of that: the
snapshot both lanes read from, the analysis lane, the answer call, and the commit.
"""

import asyncio
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from cognee.infrastructure.session.feedback_detection import analyze_turn_for_session_context
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_search_models import SessionTurnSnapshot
from cognee.infrastructure.session.session_turn import (
    apply_session_turn_analysis,
    build_active_context_block_safe,
    coerce_qa_entry,
    compose_session_prompt,
    load_served_context_payload,
    select_session_history,
)
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.session_lifecycle import track_session_usage
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_latency_turn")

# The analysis runs alongside the answer, so it normally finishes first. This only bounds
# the pathological case where it would hold the turn open past its own answer.
ANALYSIS_TIMEOUT_SECONDS = 30.0


async def load_latency_turn_snapshot(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    raw_message: str,
) -> SessionTurnSnapshot:
    """Read every piece of session state one latency turn needs, in one pass."""
    auto_feedback = session_manager.is_auto_feedback_enabled()
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
            )
        )
    loaded = await asyncio.gather(*loads)
    recent_entries, completion_history = loaded[:2]
    active_context, active_context_ids = loaded[2] if auto_feedback else ("", [])

    recent_rows = [coerce_qa_entry(entry) for entry in recent_entries or []]
    recent_qas = tuple(
        (
            str(row.get("qa_id") or ""),
            str(row.get("question") or ""),
            str(row.get("answer") or ""),
        )
        for row in recent_rows[-2:]
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

    return SessionTurnSnapshot(
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


async def analyze_latency_turn(snapshot: SessionTurnSnapshot) -> SessionTurnAnalysis:
    """Run the turn analysis alongside the answer. Fail open to no context updates.

    Latency mode uses only the two context-maintenance outputs; the routing fields are
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
        logger.warning("Latency turn analysis failed open: %s", error)
        return SessionTurnAnalysis()


async def complete_latency_turn(
    *,
    snapshot: SessionTurnSnapshot,
    context: Any,
    user_id: Any,
    session_id: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: str | None,
    response_model: type,
) -> Any:
    """Generate the turn's answer with the caller's own prompts and response model."""
    completion_call = generate_completion(
        query=snapshot.raw_message,
        context=context,
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        conversation_history=compose_session_prompt(
            snapshot.active_context,
            snapshot.completion_history,
        ),
        response_model=response_model,
    )
    if isinstance(user_id, UUID):
        async with track_session_usage(session_id, user_id):
            return await completion_call
    return await completion_call


async def commit_latency_turn(
    session_manager,
    *,
    snapshot: SessionTurnSnapshot,
    analysis: SessionTurnAnalysis,
    answer: Any,
    user_id: str,
    session_id: str,
    used_graph_element_ids: dict | None,
) -> None:
    """Apply the turn's context updates, then store the QA pair. Both fail open."""
    await apply_session_turn_analysis(
        session_manager,
        user_id=user_id,
        session_id=session_id,
        query=snapshot.raw_message,
        analysis=analysis,
        previous_qa_id=snapshot.previous_qa_id,
        served_ids=[entry_id for entry_id, _content in snapshot.previous_served_context],
    )
    await session_manager.add_qa(
        user_id=user_id,
        question=snapshot.raw_message,
        context="",
        answer=answer.model_dump_json() if isinstance(answer, BaseModel) else str(answer),
        session_id=session_id,
        used_graph_element_ids=used_graph_element_ids,
        used_session_context_ids=list(snapshot.active_context_ids) or None,
    )
