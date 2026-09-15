"""Decide, without an LLM call or a pipeline run, whether a session has anything to bridge.

Every ``improve(session_ids=...)`` stage is guarded by its own watermark or applied
marker, so a redundant improve was already *safe* — but it was not free: each stage
still started a memify pipeline (writing ``pipeline_runs`` rows), distillation still
ran its curator LLM pass, and enrichment always re-ran (SDK-593). This module reads
those same watermarks up front and reports, per stage, whether anything lies above
them. ``improve()`` uses the answer to return immediately when nothing is pending and
to skip the stages that have nothing to do.

The probe reads the session cache only: the QA entries, the trace-step count, and the
session-context rows (which hold all the watermarks). It never writes.
"""

from dataclasses import dataclass

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.session.agent_context_extraction import read_processed_trace_count
from cognee.infrastructure.session.session_context_builder import coerce_active_context_entries
from cognee.infrastructure.session.session_distillation_watermark import (
    filter_distillable_entries,
    read_distillation_watermark,
)
from cognee.infrastructure.session.session_persist_watermark import (
    SESSION_PERSIST_STATE_ID,
    SESSION_PERSIST_STATE_KIND,
)
from cognee.infrastructure.session.session_trace_persist_watermark import (
    read_persisted_trace_count,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_pending_work")


@dataclass(frozen=True, slots=True)
class SessionPendingWork:
    """Per-stage "is there anything above the watermark" answer for one session."""

    session_id: str
    new_qa: bool = False
    feedback_qas: bool = False
    new_traces_to_persist: bool = False
    new_traces_for_agent_context: bool = False
    distillable_entries: bool = False

    @property
    def any(self) -> bool:
        return (
            self.new_qa
            or self.feedback_qas
            or self.new_traces_to_persist
            or self.new_traces_for_agent_context
            or self.distillable_entries
        )

    def stages(self) -> list[str]:
        """Names of the stages with pending work (for logs and responses)."""
        names = []
        if self.feedback_qas:
            names.append("feedback_weights")
        if self.new_qa:
            names.append("persist_sessions")
        if self.new_traces_to_persist:
            names.append("persist_trace_steps")
        if self.new_traces_for_agent_context:
            names.append("extract_agent_context")
        if self.distillable_entries:
            names.append("distill_sessions")
        return names


def _read_persisted_qa_count(raw_entries: list) -> int:
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        if (
            raw.get("id") == SESSION_PERSIST_STATE_ID
            or raw.get("kind") == SESSION_PERSIST_STATE_KIND
        ):
            try:
                return max(0, int(raw.get("persisted_qa_count") or 0))
            except (TypeError, ValueError):
                return 0
    return 0


def _has_eligible_feedback(entries: list) -> bool:
    # Imported lazily: the eligibility rule lives with the task that consumes it.
    from cognee.tasks.memify.extract_feedback_qas import _is_eligible

    return any(isinstance(entry, SessionQAEntry) and _is_eligible(entry) for entry in entries)


async def probe_session_pending_work(
    session_manager, user_id: str, session_id: str
) -> SessionPendingWork:
    """Read the session's watermarks and report which improve stages have work.

    Fail-closed on the side of doing work: if the cache cannot be read, every stage
    is reported as pending so ``improve()`` falls back to its guarded stages rather
    than skipping a bridge it could not verify.
    """
    try:
        qa_entries = await session_manager.get_session(
            user_id=user_id, session_id=session_id, formatted=False
        )
        qa_entries = list(qa_entries) if isinstance(qa_entries, list) else []
        trace_count = await session_manager.get_agent_trace_count(
            user_id=user_id, session_id=session_id
        )
        context_rows = await session_manager.get_session_context_entries(
            user_id=user_id, session_id=session_id
        )
    except Exception as error:
        logger.warning(
            "pending-work probe for session '%s' failed (%s); assuming every stage is pending",
            session_id,
            error,
            exc_info=True,
        )
        return SessionPendingWork(
            session_id=session_id,
            new_qa=True,
            feedback_qas=True,
            new_traces_to_persist=True,
            new_traces_for_agent_context=True,
            distillable_entries=True,
        )

    persisted_qa = _read_persisted_qa_count(context_rows)
    # A watermark above the current count means the session was cleared and
    # rebuilt; the extractor then persists from the start again.
    new_qa = len(qa_entries) > persisted_qa or (persisted_qa > len(qa_entries) > 0)

    persisted_traces = read_persisted_trace_count(context_rows)
    new_traces_to_persist = trace_count > persisted_traces or (persisted_traces > trace_count > 0)

    auto_feedback = getattr(session_manager, "is_auto_feedback_enabled", lambda: True)()
    new_traces_for_agent_context = bool(auto_feedback) and trace_count > read_processed_trace_count(
        context_rows
    )

    distillable = filter_distillable_entries(
        coerce_active_context_entries(context_rows), read_distillation_watermark(context_rows)
    )

    return SessionPendingWork(
        session_id=session_id,
        new_qa=new_qa,
        feedback_qas=_has_eligible_feedback(qa_entries),
        new_traces_to_persist=new_traces_to_persist,
        new_traces_for_agent_context=new_traces_for_agent_context,
        distillable_entries=bool(distillable),
    )


async def probe_sessions_pending_work(
    session_manager, user_id: str, session_ids: list[str]
) -> dict[str, SessionPendingWork]:
    """``probe_session_pending_work`` for every session, keyed by session id."""
    pending: dict[str, SessionPendingWork] = {}
    for session_id in session_ids:
        pending[session_id] = await probe_session_pending_work(
            session_manager, user_id=user_id, session_id=session_id
        )
    return pending
