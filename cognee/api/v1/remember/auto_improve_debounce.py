"""Debounce for the automatic session-to-graph improve (plan item B6).

``remember(session_id=...)`` bridges the session into the permanent graph by
launching ``improve(session_ids=[session_id])`` in the background after every
call. A chatty agent that remembers after each turn therefore pays one full
improve run per turn. ``IMPROVE_DEBOUNCE_ENTRIES`` / ``IMPROVE_DEBOUNCE_SECONDS``
(``ImproveConfig``) let the bridge fire only when enough new session entries
have accumulated since the last automatic improve, **or** enough time has
passed since it.

The per-session bookkeeping — the Q&A count and the timestamp at which the
last automatic improve fired — is stored through ``StateRowWatermark``
(``session_persist_watermark``), the same internal non-rendered
session-context row every improve-stage watermark uses; the two fields live
as a dict value the way the distill watermark stores its id list. It is not
a new store: the row lives in the session cache next to the entries it
counts, so clearing the session clears it too.

Semantics of the two knobs:

* ``debounce_entries = N`` fires when at least ``N`` entries were added since
  the last automatic improve; ``0`` turns the entry trigger off.
* ``debounce_seconds = T`` fires when at least ``T`` seconds elapsed since the
  last automatic improve; ``0`` turns the time trigger off.
* A seconds-only configuration is time-only: the entries default of ``1``
  fires on every call, which would silently defeat the time knob, so with
  ``debounce_seconds > 0`` it steps aside unless set explicitly (``>= 2`` to
  combine both triggers, ``0`` for time-only spelled out).
* Both at their "off" value, or the defaults (``1`` entry, ``0`` seconds),
  mean no debounce: every ``remember()`` bridges, exactly as before. The
  default path reads nothing from the cache.
* A session with no state row yet always fires (first run).
* Reading the state failing is fail-open: the improve fires.
* There is no timer: the decision runs only inside ``remember()``, so entries
  below the thresholds wait for the *next* call. They are never dropped —
  recall reads them from the cache, and the persist stages' own watermarks
  are independent of this row, so whichever improve runs next bridges them —
  but "in the graph within T seconds" is not a promise the seconds knob makes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from cognee.infrastructure.session.session_persist_watermark import StateRowWatermark
from cognee.modules.improve.config import get_improve_config
from cognee.shared.logging_utils import get_logger

logger = get_logger("auto_improve_debounce")

AUTO_IMPROVE_STATE_ID = "auto_improve_debounce"
AUTO_IMPROVE_STATE_KIND = "auto_improve_debounce_state"

# One state row per (user, session), holding {"qa_count", "last_improve_at"}.
AUTO_IMPROVE_WATERMARK = StateRowWatermark(
    state_id=AUTO_IMPROVE_STATE_ID,
    state_kind=AUTO_IMPROVE_STATE_KIND,
    field="state",
)

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
    elapsed_seconds: float | None = None


def auto_improve_enabled() -> bool:
    """``IMPROVE_AUTO_ENABLED`` — the kill switch for both auto-improve paths."""
    return get_improve_config().auto_enabled


def debounce_active() -> bool:
    """True when the configured thresholds can ever hold an improve back."""
    config = get_improve_config()
    return config.debounce_entries > 1 or config.debounce_seconds > 0


def _parse_timestamp(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


async def _count_qa_entries(session_manager, user_id: str, session_id: str) -> int:
    entries = await session_manager.get_session(user_id=user_id, session_id=session_id)
    return len(entries) if entries else 0


async def should_auto_improve(
    session_manager,
    user_id: str,
    session_id: str,
    *,
    now: datetime | None = None,
) -> AutoImproveDecision:
    """Decide whether this ``remember()`` call should launch the improve bridge.

    Called after the new entry was written, so ``qa_count`` includes it.
    """
    config = get_improve_config()
    entries_threshold = max(0, int(config.debounce_entries))
    seconds_threshold = max(0.0, float(config.debounce_seconds))

    # A seconds-only configuration means "time is the threshold". The entries
    # default of 1 fires on every call — each remember() adds an entry — which
    # would silently defeat the time knob, so it steps aside unless the caller
    # raised it (>= 2, combine both) or turned it off themselves (0).
    if seconds_threshold > 0 and entries_threshold == 1:
        entries_threshold = 0

    if entries_threshold <= 1 and seconds_threshold <= 0:
        return AutoImproveDecision(due=True, reason=REASON_NO_DEBOUNCE)

    try:
        state = await AUTO_IMPROVE_WATERMARK.read_value(session_manager, user_id, session_id)
        qa_count = await _count_qa_entries(session_manager, user_id, session_id)
    except Exception as exc:
        logger.debug("auto-improve debounce: state unavailable, firing (%s)", exc, exc_info=True)
        return AutoImproveDecision(due=True, reason=REASON_STATE_UNAVAILABLE)

    if not isinstance(state, dict):  # no row yet, or a malformed value: fire
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
    qa_count: int | None = None,
    now: datetime | None = None,
) -> None:
    """Record that an automatic improve was launched for this session.

    Written before the improve runs, so back-to-back ``remember()`` calls see
    the advanced watermark. A launched bridge that then loses its lock claim
    gets the window refunded (``rearm_auto_improve_debounce``), so its entries
    never wait out a window behind a bridge that did nothing. Never raises:
    losing the row only means the next call fires one improve earlier than
    the thresholds ask for.
    """
    try:
        if qa_count is None:
            qa_count = await _count_qa_entries(session_manager, user_id, session_id)
        await AUTO_IMPROVE_WATERMARK.write_value(
            session_manager,
            user_id,
            session_id,
            {
                "qa_count": max(0, int(qa_count)),
                "last_improve_at": (now or datetime.now(timezone.utc)).isoformat(),
            },
        )
    except Exception as exc:
        logger.debug("auto-improve debounce: could not save state (%s)", exc, exc_info=True)


async def rearm_auto_improve_debounce(session_manager, user_id: str, session_id: str) -> None:
    """Refund the debounce budget after a bridge that did no work.

    A launched bridge that lost its improve-lock claim persisted nothing, but
    ``mark_auto_improve_fired`` already spent the window — without a refund the
    session's entries wait out a full extra debounce window behind a bridge
    that never ran. Clearing the state value makes the NEXT ``remember()`` fire
    unconditionally (a non-dict state is the first-run path). Never raises.
    """
    try:
        await AUTO_IMPROVE_WATERMARK.write_value(session_manager, user_id, session_id, None)
    except Exception as exc:
        logger.debug("auto-improve debounce: could not re-arm (%s)", exc, exc_info=True)
