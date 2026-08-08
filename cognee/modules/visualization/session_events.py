"""Collect memory-operation events from the session layer for the Memory tab.

The "operation bench": searches run against the backend (with a session)
record which graph elements produced each answer (``used_graph_element_ids``),
and feedback entries record how those answers were rated. This module turns
that history into timeline events for the visualization, so backend activity
appears on the Memory tab automatically — no hand-built ``search_events``.

Two event kinds are emitted per the renderer's contract:

- ``search``  — one per QA entry: the query, answer, and the graph elements
  retrieved to produce it (spotlight overlay).
- ``improve`` — one per *rated* QA entry: the same elements plus the rating,
  representing the reinforcement that ``improve()`` applies to them via
  ``apply_feedback_weights`` (reinforcement overlay).
"""

from typing import Any, Dict, List, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("visualization.session_events")

# Bound the number of sessions scanned when none are specified explicitly.
MAX_SESSIONS_SCANNED = 10


def map_session_entries_to_events(session_id: str, entries: List[Any]) -> List[Dict[str, Any]]:
    """Map SessionQAEntry records to Memory-tab timeline events (pure).

    Emits a ``search`` event per entry and, when the entry carries a feedback
    score, an ``improve`` event immediately after it (the renderer's sort is
    stable, so equal timestamps preserve this order).
    """
    events: List[Dict[str, Any]] = []
    for entry in entries:
        used = getattr(entry, "used_graph_element_ids", None) or {}
        node_ids = list(used.get("node_ids") or [])
        edge_ids = list(used.get("edge_ids") or [])
        question = getattr(entry, "question", "") or ""
        if not question.strip() and not node_ids:
            # Placeholder/system entries with neither a query nor provenance
            # add nothing to the timeline story.
            continue

        base = {
            "session_id": session_id,
            "qa_id": getattr(entry, "qa_id", None),
            "time": getattr(entry, "time", "") or "",
            "question": question,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
        }
        events.append({**base, "kind": "search", "answer": getattr(entry, "answer", "") or ""})

        score = getattr(entry, "feedback_score", None)
        if score is not None:
            memify_metadata = getattr(entry, "memify_metadata", None) or {}
            events.append(
                {
                    **base,
                    "kind": "improve",
                    "rating": score,
                    "feedback_text": getattr(entry, "feedback_text", None),
                    "applied": bool(memify_metadata.get("feedback_weights_applied")),
                }
            )
    return events


async def _list_recent_session_ids(user_uuid, limit: int) -> List[str]:
    """Most recently active session ids for a user, from the lifecycle table.

    ``user_uuid`` must be the raw UUID (the column is UUID-typed; a string
    fails the SQLAlchemy bind with "'str' object has no attribute 'hex'").
    """
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.models import SessionRecord

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(SessionRecord)
                    .where(SessionRecord.user_id == user_uuid)
                    .order_by(SessionRecord.last_activity_at.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
    return [str(row.session_id) for row in rows]


async def collect_session_events(
    user=None,
    session_ids: Optional[List[str]] = None,
    max_sessions: int = MAX_SESSIONS_SCANNED,
) -> List[Dict[str, Any]]:
    """Best-effort collection of search/improve events from the session cache.

    Never raises: any unavailable layer (cache disabled, no lifecycle table,
    no sessions) degrades to an empty list so visualization always renders.
    """
    try:
        from cognee.infrastructure.session.get_session_manager import get_session_manager
        from cognee.modules.users.methods import get_default_user

        if user is None:
            user = await get_default_user()
        user_id = str(user.id)

        session_manager = get_session_manager()
        if not session_manager.is_available:
            logger.debug("Session cache unavailable; no operation events collected.")
            return []

        if session_ids is None:
            try:
                session_ids = await _list_recent_session_ids(user.id, max_sessions)
            except Exception as error:  # noqa: BLE001 — lifecycle table may not exist
                logger.debug("Session listing unavailable (%s); no events collected.", error)
                return []

        events: List[Dict[str, Any]] = []
        for session_id in session_ids[:max_sessions]:
            entries = await session_manager.get_session(user_id=user_id, session_id=session_id)
            if isinstance(entries, list) and entries:
                events.extend(map_session_entries_to_events(session_id, entries))

        events.sort(key=lambda event: (event.get("time") or "", event["kind"] == "improve"))
        if events:
            logger.info("Collected %d session operation events for the Memory tab.", len(events))
        return events
    except Exception as error:  # noqa: BLE001 — visualization must never fail on this
        logger.warning("Session event collection failed; rendering without them: %s", error)
        return []
