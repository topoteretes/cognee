"""Idle-session consolidation: bridge session memory before the cache expires it.

Session memory (Q&A turns, feedback, context lessons) lives in the session
cache under ``SESSION_TTL_SECONDS`` (7 days by default). Only ``remember()``
bridges a session into the permanent graph, by running ``improve()`` on it.
Two common cases therefore lose memory for good when the TTL runs out:

* an agent that converses through ``recall()``/``search()`` with a
  ``session_id`` — those turns are written to the session but never bridged;
* a ``remember()`` session whose last entries were held back by the debounce
  (``IMPROVE_DEBOUNCE_*``), which has no timer and waits for a next call
  that may never come.

``consolidate_idle_sessions()`` closes that gap. It finds sessions that have
been idle for ``IMPROVE_IDLE_CONSOLIDATION_AFTER_SECONDS`` and are still
inside the cache TTL, keeps those holding Q&A entries past the persist
watermark, and runs the same ``improve(dataset, session_ids=[...])`` call
``remember()`` runs, into the dataset the session is attributed to. A session
with no dataset attribution is reported as ``unattributed`` and left alone:
there is no single dataset to bridge it into, and guessing one could write
memory where different permissions apply. Every improve stage is watermark-gated, so a session is
never bridged twice, and the per-session improve lock keeps concurrent
sweepers (several API workers) from doubling work.

The API server runs a sweep every ``IMPROVE_IDLE_CONSOLIDATION_INTERVAL_SECONDS``
when ``IMPROVE_IDLE_CONSOLIDATION_ENABLED=true``. SDK users and schedulers can
call ``consolidate_idle_sessions()`` directly. Like the automatic improve after
``remember()``, each bridge first asks the host admission check
(``register_auto_improve_admission``) and is skipped when it declines.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID

from cognee.shared.logging_utils import get_logger

from .config import get_improve_config

logger = get_logger("idle_consolidation")

OUTCOME_CONSOLIDATED = "consolidated"
OUTCOME_NOTHING_PENDING = "nothing_pending"
OUTCOME_UNATTRIBUTED = "unattributed"
OUTCOME_ADMISSION_DECLINED = "admission_declined"
OUTCOME_LOCK_HELD = "lock_held"
OUTCOME_ERRORED = "errored"


@dataclass
class IdleConsolidationReport:
    """What one sweep did, per outcome, with the sessions it bridged."""

    sessions_considered: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    consolidated: list[tuple[str, str]] = field(default_factory=list)

    def record(self, outcome: str) -> None:
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1


async def _pending_qa_count(session_manager, user_id: str, session_id: str) -> int:
    """Q&A entries in the session that the persist stage has not bridged yet."""
    from cognee.infrastructure.session.session_persist_watermark import get_persisted_qa_count

    entries = await session_manager.get_session(user_id=user_id, session_id=session_id)
    total = len(entries) if entries else 0
    persisted = await get_persisted_qa_count(session_manager, user_id, session_id)
    return max(0, total - persisted)


async def _consolidate_session(
    session_manager, user_id: UUID, session_id: str, dataset_id: UUID | None
) -> str:
    from cognee.api.v1.improve import improve
    from cognee.modules.improve.admission import auto_improve_skip_reason
    from cognee.modules.operations.origin import ORIGIN_BACKGROUND, operation_origin_scope
    from cognee.modules.users.methods import get_user

    if await _pending_qa_count(session_manager, str(user_id), session_id) == 0:
        return OUTCOME_NOTHING_PENDING

    if dataset_id is None:
        # A session with no dataset attribution (e.g. a recall() spanning
        # several datasets) has no single home in the graph. Guessing one
        # would write the user's memory into a dataset with its own,
        # possibly different, permissions — leave it to an explicit improve().
        return OUTCOME_UNATTRIBUTED

    user = await get_user(user_id)

    skip_reason = await auto_improve_skip_reason(
        user=user,
        dataset_id=dataset_id,
        session_id=session_id,
        session_ids=[session_id],
    )
    if skip_reason:
        logger.info(
            "idle consolidation: session '%s' skipped by the host admission check (%s)",
            session_id,
            skip_reason,
        )
        return OUTCOME_ADMISSION_DECLINED

    with operation_origin_scope(ORIGIN_BACKGROUND):
        result = await improve(dataset=dataset_id, session_ids=[session_id], user=user)

    if result.lock_held:
        # Another improve holds this session; it bridges the entries itself.
        return OUTCOME_LOCK_HELD
    if result.status == "errored":
        return OUTCOME_ERRORED
    return OUTCOME_CONSOLIDATED


async def consolidate_idle_sessions(now: datetime | None = None) -> IdleConsolidationReport:
    """Bridge idle sessions that still hold unpersisted Q&A into the graph.

    Runs one sweep regardless of ``IMPROVE_IDLE_CONSOLIDATION_ENABLED`` (that
    flag only controls the API server's periodic loop). Never raises for a
    single session: failures are logged and counted as ``errored``.
    """
    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.session.get_session_manager import get_session_manager
    from cognee.modules.session_lifecycle.metrics import list_idle_sessions

    report = IdleConsolidationReport()

    session_manager = get_session_manager()
    if not session_manager.is_available:
        return report

    config = get_improve_config()
    now = now or datetime.now(timezone.utc)
    idle_before = now - timedelta(seconds=config.idle_consolidation_after_seconds)

    session_ttl = get_cache_config().session_ttl_seconds
    active_after = now - timedelta(seconds=session_ttl) if session_ttl else None

    sessions = await list_idle_sessions(
        idle_before=idle_before,
        active_after=active_after,
        limit=config.idle_consolidation_batch_size,
    )
    report.sessions_considered = len(sessions)

    for user_id, session_id, dataset_id in sessions:
        try:
            outcome = await _consolidate_session(session_manager, user_id, session_id, dataset_id)
        except Exception as error:
            logger.warning(
                "idle consolidation: session '%s' for user %s failed (non-fatal): %s",
                session_id,
                user_id,
                error,
                exc_info=True,
            )
            outcome = OUTCOME_ERRORED

        report.record(outcome)
        if outcome == OUTCOME_CONSOLIDATED:
            report.consolidated.append((str(user_id), session_id))

    if report.consolidated:
        logger.info(
            "idle consolidation: bridged %d of %d idle session(s) into long-term memory",
            len(report.consolidated),
            report.sessions_considered,
        )
    return report


async def run_idle_consolidation_loop(stop_event: asyncio.Event) -> None:
    """Sweep every ``idle_consolidation_interval_seconds`` until ``stop_event`` is set."""
    interval = get_improve_config().idle_consolidation_interval_seconds
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass

        try:
            await consolidate_idle_sessions()
        except Exception as error:
            logger.warning("idle consolidation: sweep failed (non-fatal): %s", error, exc_info=True)
