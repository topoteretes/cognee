"""Per-session watermark for persisting agent trace steps into the knowledge graph.

Trace persistence (``improve(session_ids=...)`` -> ``extract_agent_trace_feedbacks``
-> ``cognify_agent_trace_feedback``) used to serialize EVERY stored trace step on
every run into one text blob. Any new step changed the blob, so the add-level
content-hash dedup never matched and the whole trace history was re-embedded and
re-extracted each improve — O(n^2) ingestion work for trace-heavy plugin sessions
(SDK-593).

This module mirrors ``session_persist_watermark``: a count watermark per
(user, session) stored as an internal non-rendered session-context row. The
extractor reads it and yields only the steps above it; ``cognify_agent_trace_feedback``
advances it only after that window was successfully cognified, so a failed
cognify retries the same window on the next ``improve()``.

A watermark larger than the session's current step count means the session was
cleared and rebuilt; treat it as stale and persist from the beginning again.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from cognee.shared.logging_utils import get_logger

logger = get_logger("session_trace_persist_watermark")

TRACE_PERSIST_STATE_ID = "session_trace_persist_watermark"
TRACE_PERSIST_STATE_KIND = "session_trace_persist_watermark_state"


@dataclass(frozen=True, slots=True)
class TracePersistWindow:
    """One not-yet-persisted slice of a session's trace steps.

    ``persisted_trace_count`` is the TOTAL step count captured at extraction
    time — the value the watermark advances to once this window is
    successfully cognified.
    """

    user_id: str
    session_id: str
    text: str
    persisted_trace_count: int


def _extract_state_row(raw_entries: list) -> dict | None:
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("id") == TRACE_PERSIST_STATE_ID:
            return raw
        if raw.get("kind") == TRACE_PERSIST_STATE_KIND:
            return raw
    return None


def read_persisted_trace_count(raw_entries: list) -> int:
    """Read the watermark out of already-loaded context rows (0 when absent/malformed)."""
    row = _extract_state_row(raw_entries)
    if row is None:
        return 0
    try:
        return max(0, int(row.get("persisted_trace_count") or 0))
    except (TypeError, ValueError):
        return 0


async def get_persisted_trace_count(session_manager, user_id: str, session_id: str) -> int:
    """Read the trace persist watermark. Missing or malformed state means nothing persisted."""
    raw_entries = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id
    )
    return read_persisted_trace_count(raw_entries)


async def save_persisted_trace_count(
    session_manager, user_id: str, session_id: str, persisted_trace_count: int
) -> None:
    """Persist the watermark as an internal non-rendered session-context row."""
    payload = {
        "id": TRACE_PERSIST_STATE_ID,
        "kind": TRACE_PERSIST_STATE_KIND,
        "persisted_trace_count": max(0, int(persisted_trace_count)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    updated = await session_manager.update_session_context_entry(
        user_id=user_id,
        session_id=session_id,
        entry_id=TRACE_PERSIST_STATE_ID,
        merge=payload,
    )
    if not updated:
        await session_manager.create_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_dump=payload,
        )
