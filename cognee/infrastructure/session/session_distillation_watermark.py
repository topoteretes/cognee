"""Per-session watermark for session distillation.

``distill_session`` used to run the curator LLM pass over ALL of a session's
gated guidance entries and QA turns on every ``improve()``, re-proposing (and
re-judging) the same lessons each time (SDK-593). This module stores, per
(user, session), the newest ``created_at`` / ``time`` stamp the curator has
already consumed — an internal non-rendered session-context row, the same
policy as the persist and agent-context watermarks. Distillation feeds the
curator only entries and QA turns strictly newer than the watermark and
advances it once the curator pass has completed (any terminal status): the
entries were evaluated, so re-evaluating them next time would only spend LLM
calls. An exception leaves the watermark untouched, so the same window is
retried on the next ``improve()``.

Timestamps are ISO-8601 strings produced by ``datetime.now(timezone.utc).isoformat()``
on both the QA and the context-entry write paths, so the string comparison the
curator batching already relies on is used here as well.
"""

from datetime import datetime, timezone

from cognee.infrastructure.session.session_context_models import (
    SessionContextEntry,
    is_context_entry_usable,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_distillation_watermark")

SESSION_DISTILLATION_STATE_ID = "session_distillation_watermark"
SESSION_DISTILLATION_STATE_KIND = "session_distillation_watermark_state"


def _extract_state_row(raw_entries: list) -> dict | None:
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("id") == SESSION_DISTILLATION_STATE_ID:
            return raw
        if raw.get("kind") == SESSION_DISTILLATION_STATE_KIND:
            return raw
    return None


def read_distillation_watermark(raw_entries: list) -> str:
    """Read the watermark out of already-loaded context rows ('' when absent/malformed)."""
    row = _extract_state_row(raw_entries)
    if row is None:
        return ""
    value = row.get("distilled_through")
    return value if isinstance(value, str) else ""


def is_newer_than_watermark(stamp: str | None, watermark: str) -> bool:
    """True when ``stamp`` lies strictly after ``watermark`` (an empty watermark matches all)."""
    if not watermark:
        return True
    return bool(stamp) and stamp > watermark


def filter_distillable_entries(
    entries: list[SessionContextEntry], watermark: str
) -> list[SessionContextEntry]:
    """Keep usable entries the curator has not consumed yet."""
    return [
        entry
        for entry in entries
        if is_context_entry_usable(entry) and is_newer_than_watermark(entry.created_at, watermark)
    ]


def filter_distillable_qa_rows(qa_rows: list[dict], watermark: str) -> list[dict]:
    """Keep QA turns the curator has not consumed yet."""
    return [row for row in qa_rows if is_newer_than_watermark(row.get("time"), watermark)]


def next_distillation_watermark(
    qa_rows: list[dict], entries: list[SessionContextEntry], current: str
) -> str:
    """The newest stamp among the consumed inputs (never moves backwards)."""
    stamps = [current]
    stamps.extend(row.get("time") or "" for row in qa_rows)
    stamps.extend(entry.created_at or "" for entry in entries)
    return max(stamps)


async def save_distillation_watermark(
    session_manager, user_id: str, session_id: str, distilled_through: str
) -> None:
    """Persist the watermark as an internal non-rendered session-context row."""
    payload = {
        "id": SESSION_DISTILLATION_STATE_ID,
        "kind": SESSION_DISTILLATION_STATE_KIND,
        "distilled_through": distilled_through,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    updated = await session_manager.update_session_context_entry(
        user_id=user_id,
        session_id=session_id,
        entry_id=SESSION_DISTILLATION_STATE_ID,
        merge=payload,
    )
    if not updated:
        await session_manager.create_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_dump=payload,
        )
