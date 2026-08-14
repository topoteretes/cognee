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

# Same policy for agent trace steps: improve() previously re-extracted and
# re-cognified the FULL growing trace blob per run (O(n^2) work per session).
TRACE_PERSIST_STATE_ID = "trace_persist_watermark"
TRACE_PERSIST_STATE_KIND = "trace_persist_watermark_state"


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


@dataclass(frozen=True, slots=True)
class TracePersistWindow:
    """One not-yet-persisted slice of a session's agent trace steps.

    ``persisted_trace_count`` is the TOTAL step count captured at extraction
    time — the value the watermark advances to once this window is
    successfully cognified.
    """

    user_id: str
    session_id: str
    text: str
    persisted_trace_count: int


async def load_state_row(
    session_manager, user_id: str, session_id: str, state_id: str, state_kind: str | None = None
) -> dict | None:
    """Find an internal state row by id (or, when given, by kind as a fallback).

    Shared by every session-state consumer (persist/trace watermarks, distillation
    watermark, auto-improve debounce) so the row-scan lives in one place. Kind
    matching is opt-in: per-dataset state ids share one kind, so matching by kind
    would cross-match datasets.
    """
    raw_entries = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id
    )
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("id") == state_id:
            return raw
        if state_kind is not None and raw.get("kind") == state_kind:
            return raw
    return None


async def save_state_row(session_manager, user_id: str, session_id: str, payload: dict) -> None:
    """Upsert an internal non-rendered session-context row keyed by ``payload["id"]``."""
    updated = await session_manager.update_session_context_entry(
        user_id=user_id,
        session_id=session_id,
        entry_id=payload["id"],
        merge=payload,
    )
    if not updated:
        await session_manager.create_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_dump=payload,
        )


async def _get_count(
    session_manager, user_id: str, session_id: str, state_id: str, state_kind: str, field: str
) -> int:
    """Read a count watermark row. Missing or malformed state means zero."""
    row = await load_state_row(session_manager, user_id, session_id, state_id, state_kind)
    if row is None:
        return 0
    try:
        return max(0, int(row.get(field) or 0))
    except (TypeError, ValueError):
        return 0


async def _save_count(
    session_manager,
    user_id: str,
    session_id: str,
    state_id: str,
    state_kind: str,
    field: str,
    value: int,
) -> None:
    """Persist a count watermark as an internal non-rendered session-context row."""
    await save_state_row(
        session_manager,
        user_id,
        session_id,
        {
            "id": state_id,
            "kind": state_kind,
            field: max(0, int(value)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


async def get_persisted_qa_count(session_manager, user_id: str, session_id: str) -> int:
    """Read the persist watermark. Missing or malformed state means nothing persisted yet."""
    return await _get_count(
        session_manager,
        user_id,
        session_id,
        SESSION_PERSIST_STATE_ID,
        SESSION_PERSIST_STATE_KIND,
        "persisted_qa_count",
    )


async def save_persisted_qa_count(
    session_manager, user_id: str, session_id: str, persisted_qa_count: int
) -> None:
    """Persist the watermark as an internal non-rendered session-context row."""
    await _save_count(
        session_manager,
        user_id,
        session_id,
        SESSION_PERSIST_STATE_ID,
        SESSION_PERSIST_STATE_KIND,
        "persisted_qa_count",
        persisted_qa_count,
    )


async def get_persisted_trace_count(session_manager, user_id: str, session_id: str) -> int:
    """Read the trace persist watermark. Missing/malformed means nothing persisted yet."""
    return await _get_count(
        session_manager,
        user_id,
        session_id,
        TRACE_PERSIST_STATE_ID,
        TRACE_PERSIST_STATE_KIND,
        "persisted_trace_count",
    )


async def save_persisted_trace_count(
    session_manager, user_id: str, session_id: str, persisted_trace_count: int
) -> None:
    """Persist the trace watermark as an internal non-rendered session-context row."""
    await _save_count(
        session_manager,
        user_id,
        session_id,
        TRACE_PERSIST_STATE_ID,
        TRACE_PERSIST_STATE_KIND,
        "persisted_trace_count",
        persisted_trace_count,
    )
