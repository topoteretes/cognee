"""One watermark implementation for the self-improvement stages.

Several improve() stages read a session, do expensive work over it (cognify,
LLM extraction, distillation) and must not repeat that work on the next run.
Each of them keeps a *watermark*: a small record of how far it already got,
stored as an internal non-rendered session-context row per (user, session).
Context-row consumers keep only rows whose ``kind`` is ``"context"`` (or
``"feedback"``), so these state rows never reach a prompt.

This module holds the single implementation of that row (``StateRowWatermark``)
and the concrete watermarks built on it:

* ``SESSION_PERSIST_WATERMARK`` — stage 2, persisted Q&A entry count;
* ``TRACE_PERSIST_WATERMARK`` — stage 3, persisted agent-trace step count;
* ``TRACE_EXTRACTION_WATERMARK`` (declared in ``agent_context_extraction``) —
  stage 4, trace steps already turned into agent lessons;
* ``distill_watermark(dataset_id)`` — stage 5, gated context-entry ids already
  distilled into that dataset.

Two policies apply to every count watermark:

1. **A watermark above the current total means the session was cleared and
   rebuilt** (e.g. ``forget`` semantics that keep context rows). It is stale,
   and the stage starts over from the beginning (``resolve_effective``).
2. **A retried window is harmless.** The watermark is read before the work and
   advanced only after the window was committed; a failed run leaves it
   untouched so the same window is retried next time, and add-level
   content-hash dedup absorbs the repeat.

The legacy ``get_persisted_qa_count`` / ``save_persisted_qa_count`` functions
are kept as thin wrappers over ``SESSION_PERSIST_WATERMARK``.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Union
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("session_persist_watermark")

SESSION_PERSIST_STATE_ID = "session_persist_watermark"
SESSION_PERSIST_STATE_KIND = "session_persist_watermark_state"

TRACE_PERSIST_STATE_ID = "agent_trace_persist_watermark"
TRACE_PERSIST_STATE_KIND = "agent_trace_persist_watermark_state"

SESSION_DISTILL_STATE_ID_PREFIX = "session_distill_watermark"
SESSION_DISTILL_STATE_KIND = "session_distill_watermark_state"
SESSION_DISTILL_FIELD = "processed_entry_ids"


@dataclass(frozen=True, slots=True)
class StateRowWatermark:
    """A watermark stored as one internal session-context row per (user, session).

    ``state_id`` is the row's ``id`` (one row per watermark per session, updated
    in place); ``state_kind`` is the row's ``kind``, which keeps it out of every
    rendered context; ``field`` is the row key holding the watermark value.

    Two value shapes are supported through one storage path: an integer count
    (``read_count`` / ``write_count``) and an opaque value such as a list of
    ids (``read_value`` / ``write_value``).

    ``match_kind`` controls the lookup fallback: when True a row is found by
    ``id`` or, failing that, by ``kind``. Watermarks whose id carries a scope
    suffix (one row per dataset, say) must set it False so a sibling row of the
    same kind is never mistaken for this one.
    """

    state_id: str
    state_kind: str
    field: str
    match_kind: bool = True

    # -- lookup ------------------------------------------------------------

    def find_row(self, raw_entries: Optional[Iterable]) -> Optional[dict]:
        """Find this watermark's state row among already-loaded context rows."""
        rows = [raw for raw in (raw_entries or []) if isinstance(raw, dict)]
        for raw in rows:
            if raw.get("id") == self.state_id:
                return raw
        if self.match_kind:
            for raw in rows:
                if raw.get("kind") == self.state_kind:
                    return raw
        return None

    def count_from_rows(self, raw_entries: Optional[Iterable]) -> int:
        """Parse the count out of already-loaded rows; missing or malformed -> 0."""
        row = self.find_row(raw_entries)
        if row is None:
            return 0
        try:
            return max(0, int(row.get(self.field) or 0))
        except (TypeError, ValueError):
            return 0

    def value_from_rows(self, raw_entries: Optional[Iterable]) -> Any:
        """Return the raw stored value, or None when the row is missing."""
        row = self.find_row(raw_entries)
        if row is None:
            return None
        return row.get(self.field)

    # -- policies ----------------------------------------------------------

    @staticmethod
    def resolve_effective(stored: int, current_total: int, *, session_id: str = "") -> int:
        """Apply the stale-watermark policy: above the current total means start over."""
        if stored > current_total:
            logger.warning(
                "Session %s has %d entries but watermark is %d; "
                "treating watermark as stale and starting from the beginning",
                session_id,
                current_total,
                stored,
            )
            return 0
        return stored

    # -- storage -----------------------------------------------------------

    async def read_row(self, session_manager, user_id: str, session_id: str) -> Optional[dict]:
        raw_entries = await session_manager.get_session_context_entries(
            user_id=user_id, session_id=session_id
        )
        return self.find_row(raw_entries)

    async def read_count(self, session_manager, user_id: str, session_id: str) -> int:
        """Read the count watermark. Missing or malformed state means nothing done yet."""
        raw_entries = await session_manager.get_session_context_entries(
            user_id=user_id, session_id=session_id
        )
        return self.count_from_rows(raw_entries)

    async def read_value(self, session_manager, user_id: str, session_id: str) -> Any:
        """Read the raw stored value; None when no row exists."""
        raw_entries = await session_manager.get_session_context_entries(
            user_id=user_id, session_id=session_id
        )
        return self.value_from_rows(raw_entries)

    async def write_value(self, session_manager, user_id: str, session_id: str, value: Any) -> None:
        """Store ``value`` under ``field``, updating the existing row or creating it."""
        payload = {
            "id": self.state_id,
            "kind": self.state_kind,
            self.field: value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        updated = await session_manager.update_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_id=self.state_id,
            merge=payload,
        )
        if not updated:
            await session_manager.create_session_context_entry(
                user_id=user_id,
                session_id=session_id,
                entry_dump=payload,
            )

    async def write_count(self, session_manager, user_id: str, session_id: str, count: int) -> None:
        """Store a count watermark (clamped at zero)."""
        await self.write_value(session_manager, user_id, session_id, max(0, int(count)))


# -- Stage 2: persisted session Q&A ------------------------------------------

SESSION_PERSIST_WATERMARK = StateRowWatermark(
    state_id=SESSION_PERSIST_STATE_ID,
    state_kind=SESSION_PERSIST_STATE_KIND,
    field="persisted_qa_count",
)


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


async def get_persisted_qa_count(session_manager, user_id: str, session_id: str) -> int:
    """Read the Q&A persist watermark. Missing or malformed state means nothing persisted yet."""
    return await SESSION_PERSIST_WATERMARK.read_count(session_manager, user_id, session_id)


async def save_persisted_qa_count(
    session_manager, user_id: str, session_id: str, persisted_qa_count: int
) -> None:
    """Persist the Q&A watermark as an internal non-rendered session-context row."""
    await SESSION_PERSIST_WATERMARK.write_count(
        session_manager, user_id, session_id, persisted_qa_count
    )


# -- Stage 3: persisted agent trace steps ------------------------------------

TRACE_PERSIST_WATERMARK = StateRowWatermark(
    state_id=TRACE_PERSIST_STATE_ID,
    state_kind=TRACE_PERSIST_STATE_KIND,
    field="persisted_trace_count",
)


@dataclass(frozen=True, slots=True)
class TracePersistWindow:
    """One not-yet-persisted slice of a session's agent trace steps.

    ``persisted_trace_count`` is the TOTAL trace-step count captured at
    extraction time — the value the watermark advances to once this window is
    successfully cognified.
    """

    user_id: str
    session_id: str
    text: str
    persisted_trace_count: int


async def get_persisted_trace_count(session_manager, user_id: str, session_id: str) -> int:
    """Read the trace persist watermark. Missing or malformed state means nothing persisted yet."""
    return await TRACE_PERSIST_WATERMARK.read_count(session_manager, user_id, session_id)


async def save_persisted_trace_count(
    session_manager, user_id: str, session_id: str, persisted_trace_count: int
) -> None:
    """Persist the trace watermark as an internal non-rendered session-context row."""
    await TRACE_PERSIST_WATERMARK.write_count(
        session_manager, user_id, session_id, persisted_trace_count
    )


# -- Stage 5: distilled context entries, per (session, dataset) --------------


def distill_watermark(dataset_id: Union[str, UUID]) -> StateRowWatermark:
    """The distillation watermark for one target dataset.

    Gated context entries are not append-only (an entry can be gated in later
    when its confidence rises, or deleted by invalidation), so a count cannot
    describe them. The row stores the ids of the gated entries that were
    already distilled into ``dataset_id``; the entries not in that set are the
    new work. The id carries the dataset so one session distilled into two
    datasets keeps two rows, hence ``match_kind=False``.
    """
    return StateRowWatermark(
        state_id=f"{SESSION_DISTILL_STATE_ID_PREFIX}:{dataset_id}",
        state_kind=SESSION_DISTILL_STATE_KIND,
        field=SESSION_DISTILL_FIELD,
        match_kind=False,
    )


async def get_distilled_entry_ids(
    session_manager, user_id: str, session_id: str, dataset_id: Union[str, UUID]
) -> set[str]:
    """Ids of the gated context entries already distilled into ``dataset_id``."""
    value = await distill_watermark(dataset_id).read_value(session_manager, user_id, session_id)
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(entry_id) for entry_id in value if entry_id}


async def save_distilled_entry_ids(
    session_manager,
    user_id: str,
    session_id: str,
    dataset_id: Union[str, UUID],
    entry_ids: Iterable[str],
) -> None:
    """Record the gated entry ids covered by a finished distillation run.

    The stored set is *replaced*, not appended to: it is the set of currently
    gated entries the run saw, so ids of entries that were since deleted drop
    out and the row stays bounded by the session's context-entry cap.
    """
    await distill_watermark(dataset_id).write_value(
        session_manager, user_id, session_id, sorted({str(entry_id) for entry_id in entry_ids})
    )
