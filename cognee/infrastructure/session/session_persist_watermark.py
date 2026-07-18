"""Per-session watermark for persisting session Q&A into the knowledge graph.

Session persistence (``improve(session_ids=...)`` -> ``extract_user_sessions``
-> ``cognify_session``) previously serialized the ENTIRE session on every run:
each time a session grew, the full history was re-added, re-embedded and
re-extracted as a brand-new document, and the previous snapshot document was
left behind — O(n^2) ingestion work for a session bridged after every entry.

This module stores a count watermark per (user, session) as an internal
non-rendered session-context row — the same policy as the agent-context
extraction watermark (see ``agent_context_extraction``): read the number of
already-persisted Q&A entries before extraction, persist only entries above
it, and advance it only after the window was successfully cognified. A failed
cognify leaves the watermark untouched, so the same window is retried on the
next ``improve()`` (add-level content-hash dedup makes that retry safe).

A watermark larger than the session's current entry count means the session
was cleared and rebuilt (e.g. ``forget`` semantics that keep context rows);
treat it as stale and persist from the beginning again.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from cognee.shared.logging_utils import get_logger

logger = get_logger("session_persist_watermark")

SESSION_PERSIST_STATE_ID = "session_persist_watermark"
SESSION_PERSIST_STATE_KIND = "session_persist_watermark_state"


@dataclass(frozen=True, slots=True)
class SessionPersistWindow:
    """One not-yet-persisted slice of a session's Q&A entries.

    ``persisted_qa_count`` is the TOTAL entry count captured at extraction
    time — the value the watermark advances to once this window is
    successfully cognified. Entries appended after extraction stay above it
    and are picked up by the next run.
    """

    user_id: str
    session_id: str
    text: str
    persisted_qa_count: int


def _extract_state_row(raw_entries: list) -> dict | None:
    """Find this session's internal persist-watermark row, if present."""
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("id") == SESSION_PERSIST_STATE_ID:
            return raw
        if raw.get("kind") == SESSION_PERSIST_STATE_KIND:
            return raw
    return None


async def get_persisted_qa_count(session_manager, user_id: str, session_id: str) -> int:
    """Read the persist watermark. Missing or malformed state means nothing persisted yet."""
    raw_entries = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id
    )
    row = _extract_state_row(raw_entries)
    if row is None:
        return 0
    try:
        return max(0, int(row.get("persisted_qa_count") or 0))
    except (TypeError, ValueError):
        return 0


async def save_persisted_qa_count(
    session_manager, user_id: str, session_id: str, persisted_qa_count: int
) -> None:
    """Persist the watermark as an internal non-rendered session-context row."""
    payload = {
        "id": SESSION_PERSIST_STATE_ID,
        "kind": SESSION_PERSIST_STATE_KIND,
        "persisted_qa_count": max(0, int(persisted_qa_count)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    updated = await session_manager.update_session_context_entry(
        user_id=user_id,
        session_id=session_id,
        entry_id=SESSION_PERSIST_STATE_ID,
        merge=payload,
    )
    if not updated:
        await session_manager.create_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_dump=payload,
        )
