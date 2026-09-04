"""Debounce for the automatic session-to-graph improve (plan item B6).

``remember(session_id=...)`` bridges the session into the permanent graph by
launching ``improve(session_ids=[session_id])`` in the background after every
call. A chatty agent that remembers after each turn therefore pays one improve
chain per turn. ``IMPROVE_DEBOUNCE_ENTRIES`` / ``IMPROVE_DEBOUNCE_SECONDS``
(``ImproveConfig``) let the bridge fire only when enough new session entries
have accumulated since the last automatic improve, **or** enough time has
passed since it.

The per-session bookkeeping — the Q&A count and the timestamp at which the
last automatic improve fired — is stored as an internal non-rendered
session-context row, the same pattern ``session_persist_watermark`` and the
agent-context extraction watermark use. It is not a new store: the row lives
in the session cache next to the entries it counts, so clearing the session
clears it too.

Semantics of the two knobs:

* ``debounce_entries = N`` fires when at least ``N`` entries were added since
  the last automatic improve; ``0`` turns the entry trigger off.
* ``debounce_seconds = T`` fires when at least ``T`` seconds elapsed since the
  last automatic improve; ``0`` turns the time trigger off.
* Both at their "off" value, or the defaults (``1`` entry, ``0`` seconds),
  mean no debounce: every ``remember()`` bridges, exactly as before. The
  default path reads nothing from the cache.
* A session with no state row yet always fires (first run).
* Reading the state failing is fail-open: the improve fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from cognee.modules.improve.config import get_improve_config
from cognee.shared.logging_utils import get_logger

logger = get_logger("auto_improve_debounce")

AUTO_IMPROVE_STATE_ID = "auto_improve_debounce"
AUTO_IMPROVE_STATE_KIND = "auto_improve_debounce_state"

REASON_NO_DEBOUNCE = "no_debounce"
REASON_FIRST_RUN = "first_run"
REASON_ENTRIES = "entries"
REASON_ELAPSED = "elapsed"
REASON_DEBOUNCED = "debounced"
REASON_STATE_UNAVAILABLE = "state_unavailable"


@dataclass(frozen=True)
class AutoImproveDecision:
    """Whether the automatic improve should fire for this ``remember()`` call."""

    due: bool
    reason: str
    qa_count: int = 0
    new_entries: int = 0
    elapsed_seconds: Optional[float] = None


def auto_improve_enabled() -> bool:
    """``IMPROVE_AUTO_ENABLED`` — the kill switch for both auto-improve paths."""
    return get_improve_config().auto_enabled


def debounce_active() -> bool:
    """True when the configured thresholds can ever hold an improve back."""
    config = get_improve_config()
    return config.debounce_entries > 1 or config.debounce_seconds > 0


def _extract_state_row(raw_entries: list) -> Optional[dict]:
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("id") == AUTO_IMPROVE_STATE_ID:
            return raw
        if raw.get("kind") == AUTO_IMPROVE_STATE_KIND:
            return raw
    return None


def _parse_timestamp(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


async def _read_state(session_manager, user_id: str, session_id: str) -> Optional[dict]:
    raw_entries = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id
    )
    return _extract_state_row(raw_entries)


async def _count_qa_entries(session_manager, user_id: str, session_id: str) -> int:
    entries = await session_manager.get_session(user_id=user_id, session_id=session_id)
    return len(entries) if entries else 0


async def should_auto_improve(
    session_manager,
    user_id: str,
    session_id: str,
    *,
    now: Optional[datetime] = None,
) -> AutoImproveDecision:
    """Decide whether this ``remember()`` call should launch the improve bridge.

    Called after the new entry was written, so ``qa_count`` includes it.
    """
    config = get_improve_config()
    entries_threshold = max(0, int(config.debounce_entries))
    seconds_threshold = max(0.0, float(config.debounce_seconds))

    if entries_threshold <= 1 and seconds_threshold <= 0:
        return AutoImproveDecision(due=True, reason=REASON_NO_DEBOUNCE)

    try:
        state = await _read_state(session_manager, user_id, session_id)
        qa_count = await _count_qa_entries(session_manager, user_id, session_id)
    except Exception as exc:
        logger.debug("auto-improve debounce: state unavailable, firing (%s)", exc)
        return AutoImproveDecision(due=True, reason=REASON_STATE_UNAVAILABLE)

    if state is None:
        return AutoImproveDecision(due=True, reason=REASON_FIRST_RUN, qa_count=qa_count)

    try:
        last_count = max(0, int(state.get("qa_count") or 0))
    except (TypeError, ValueError):
        last_count = 0
    # A watermark above the current count means the session was cleared and
    # rebuilt; every current entry is new then.
    new_entries = qa_count if qa_count < last_count else qa_count - last_count

    last_at = _parse_timestamp(state.get("last_improve_at"))
    current = now or datetime.now(timezone.utc)
    elapsed = (current - last_at).total_seconds() if last_at is not None else None

    if entries_threshold > 0 and new_entries >= entries_threshold:
        return AutoImproveDecision(
            due=True,
            reason=REASON_ENTRIES,
            qa_count=qa_count,
            new_entries=new_entries,
            elapsed_seconds=elapsed,
        )
    if seconds_threshold > 0 and (elapsed is None or elapsed >= seconds_threshold):
        return AutoImproveDecision(
            due=True,
            reason=REASON_ELAPSED,
            qa_count=qa_count,
            new_entries=new_entries,
            elapsed_seconds=elapsed,
        )
    return AutoImproveDecision(
        due=False,
        reason=REASON_DEBOUNCED,
        qa_count=qa_count,
        new_entries=new_entries,
        elapsed_seconds=elapsed,
    )


async def mark_auto_improve_fired(
    session_manager,
    user_id: str,
    session_id: str,
    *,
    qa_count: Optional[int] = None,
    now: Optional[datetime] = None,
) -> None:
    """Record that an automatic improve was launched for this session.

    Written before the improve runs, so back-to-back ``remember()`` calls see
    the advanced watermark. Never raises: losing the row only means the next
    call fires one improve earlier than the thresholds ask for.
    """
    try:
        if qa_count is None:
            qa_count = await _count_qa_entries(session_manager, user_id, session_id)
        payload = {
            "id": AUTO_IMPROVE_STATE_ID,
            "kind": AUTO_IMPROVE_STATE_KIND,
            "qa_count": max(0, int(qa_count)),
            "last_improve_at": (now or datetime.now(timezone.utc)).isoformat(),
        }
        updated = await session_manager.update_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_id=AUTO_IMPROVE_STATE_ID,
            merge=payload,
        )
        if not updated:
            await session_manager.create_session_context_entry(
                user_id=user_id,
                session_id=session_id,
                entry_dump=payload,
            )
    except Exception as exc:
        logger.debug("auto-improve debounce: could not save state (%s)", exc)
