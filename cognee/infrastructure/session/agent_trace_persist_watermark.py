"""Per-session watermark for persisting agent traces into the knowledge graph.

The extractor reads the number of successfully persisted trace steps and emits
only the append-only suffix above that count. The cognify task advances the
watermark after graph ingestion succeeds, leaving failed windows pending for a
later ``improve()`` retry.
"""

from dataclasses import dataclass
from datetime import datetime, timezone


AGENT_TRACE_PERSIST_STATE_ID = "agent_trace_persist_watermark"
AGENT_TRACE_PERSIST_STATE_KIND = "agent_trace_persist_watermark_state"


@dataclass(frozen=True, slots=True)
class AgentTracePersistWindow:
    """One not-yet-persisted slice of a session's agent trace."""

    user_id: str
    session_id: str
    text: str
    persisted_trace_count: int


def _extract_state_row(raw_entries: list) -> dict | None:
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("id") == AGENT_TRACE_PERSIST_STATE_ID:
            return raw
        if raw.get("kind") == AGENT_TRACE_PERSIST_STATE_KIND:
            return raw
    return None


async def get_persisted_trace_count(session_manager, user_id: str, session_id: str) -> int:
    """Read the trace watermark; missing or malformed state starts at zero."""
    raw_entries = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id
    )
    row = _extract_state_row(raw_entries)
    if row is None:
        return 0
    try:
        return max(0, int(row.get("persisted_trace_count") or 0))
    except (TypeError, ValueError):
        return 0


async def save_persisted_trace_count(
    session_manager, user_id: str, session_id: str, persisted_trace_count: int
) -> None:
    """Store the trace watermark as an internal, non-rendered context row."""
    payload = {
        "id": AGENT_TRACE_PERSIST_STATE_ID,
        "kind": AGENT_TRACE_PERSIST_STATE_KIND,
        "persisted_trace_count": max(0, int(persisted_trace_count)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    updated = await session_manager.update_session_context_entry(
        user_id=user_id,
        session_id=session_id,
        entry_id=AGENT_TRACE_PERSIST_STATE_ID,
        merge=payload,
    )
    if not updated:
        await session_manager.create_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_dump=payload,
        )
