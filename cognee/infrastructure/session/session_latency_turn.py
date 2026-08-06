import asyncio

from cognee.infrastructure.session.session_search_models import SessionTurnSnapshot
from cognee.infrastructure.session.session_turn import (
    build_active_context_block_safe,
    coerce_qa_entry,
    load_served_context_payload,
    select_session_history,
)


async def load_latency_turn_snapshot(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    raw_message: str,
) -> SessionTurnSnapshot:
    """Load the immutable session state used throughout one latency turn."""
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
    active_context_result = loaded[2] if auto_feedback else ("", [])

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

    active_context, active_context_ids = active_context_result
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
