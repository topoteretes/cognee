"""The improve() pending-work probe: watermarks in, per-stage booleans out, no LLM, no pipeline."""

from types import SimpleNamespace

import pytest

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.session.agent_context_extraction import (
    TRACE_EXTRACTION_STATE_ID,
    TRACE_EXTRACTION_STATE_KIND,
)
from cognee.infrastructure.session.session_distillation_watermark import (
    SESSION_DISTILLATION_STATE_ID,
    SESSION_DISTILLATION_STATE_KIND,
)
from cognee.infrastructure.session.session_persist_watermark import (
    SESSION_PERSIST_STATE_ID,
    SESSION_PERSIST_STATE_KIND,
)
from cognee.infrastructure.session.session_trace_persist_watermark import (
    TRACE_PERSIST_STATE_ID,
    TRACE_PERSIST_STATE_KIND,
)
from cognee.modules.session_bridge import (
    SessionPendingWork,
    probe_session_pending_work,
    probe_sessions_pending_work,
)
from cognee.tasks.memify.feedback_weights_constants import (
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
)


class FakeSessionManager:
    def __init__(self, *, qa=None, trace_count=0, context=None, auto_feedback=True, fail=False):
        self.qa = qa or []
        self.trace_count = trace_count
        self.context = context or []
        self.auto_feedback = auto_feedback
        self.fail = fail

    def is_auto_feedback_enabled(self):
        return self.auto_feedback

    async def get_session(self, *, user_id, session_id, formatted=False):
        if self.fail:
            raise RuntimeError("cache down")
        return list(self.qa)

    async def get_agent_trace_count(self, *, user_id, session_id):
        return self.trace_count

    async def get_session_context_entries(self, *, user_id, session_id):
        return list(self.context)


def _qa(feedback_score=None, applied=False, used=None):
    return SessionQAEntry(
        time="2026-09-09T10:00:00+00:00",
        question="q",
        context="",
        answer="a",
        qa_id="qa1",
        feedback_score=feedback_score,
        used_graph_element_ids=used,
        memify_metadata={MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY: True} if applied else None,
    )


def _entry(created_at="2026-09-09T10:00:00+00:00", confidence=0.9, harmful=0):
    return {
        "id": f"e-{created_at}",
        "section": "lessons_learned",
        "content": "A lesson.",
        "confidence": confidence,
        "created_at": created_at,
        "harmful_count": harmful,
        "kind": "context",
    }


def _persist(count):
    return {
        "id": SESSION_PERSIST_STATE_ID,
        "kind": SESSION_PERSIST_STATE_KIND,
        "persisted_qa_count": count,
    }


def _trace_persist(count):
    return {
        "id": TRACE_PERSIST_STATE_ID,
        "kind": TRACE_PERSIST_STATE_KIND,
        "persisted_trace_count": count,
    }


def _agent_ctx(count):
    return {
        "id": TRACE_EXTRACTION_STATE_ID,
        "kind": TRACE_EXTRACTION_STATE_KIND,
        "processed_trace_count": count,
    }


def _distilled(through):
    return {
        "id": SESSION_DISTILLATION_STATE_ID,
        "kind": SESSION_DISTILLATION_STATE_KIND,
        "distilled_through": through,
    }


@pytest.mark.asyncio
async def test_empty_session_has_nothing_pending():
    work = await probe_session_pending_work(FakeSessionManager(), user_id="u", session_id="s")

    assert work == SessionPendingWork(session_id="s")
    assert work.any is False
    assert work.stages() == []


@pytest.mark.asyncio
async def test_fully_bridged_session_has_nothing_pending():
    """Every counter at its watermark, feedback applied, guidance distilled: a no-op."""
    manager = FakeSessionManager(
        qa=[_qa(feedback_score=5, applied=True, used={"node_ids": ["n1"]})],
        trace_count=4,
        context=[
            _persist(1),
            _trace_persist(4),
            _agent_ctx(4),
            _entry("2026-09-09T10:00:00+00:00"),
            _distilled("2026-09-09T10:00:00+00:00"),
        ],
    )

    work = await probe_session_pending_work(manager, user_id="u", session_id="s")

    assert work.any is False


@pytest.mark.asyncio
async def test_each_watermark_flags_its_own_stage():
    base_context = [_persist(1), _trace_persist(4), _agent_ctx(4)]

    new_qa = await probe_session_pending_work(
        FakeSessionManager(qa=[_qa(), _qa()], trace_count=4, context=base_context),
        user_id="u",
        session_id="s",
    )
    assert new_qa.new_qa is True and new_qa.stages() == ["persist_sessions"]

    new_trace = await probe_session_pending_work(
        FakeSessionManager(qa=[_qa()], trace_count=5, context=base_context),
        user_id="u",
        session_id="s",
    )
    assert new_trace.new_traces_to_persist is True
    assert new_trace.new_traces_for_agent_context is True
    assert new_trace.stages() == ["persist_trace_steps", "extract_agent_context"]

    feedback = await probe_session_pending_work(
        FakeSessionManager(
            qa=[_qa(feedback_score=4, used={"edge_ids": ["e1"]})],
            trace_count=4,
            context=base_context,
        ),
        user_id="u",
        session_id="s",
    )
    assert feedback.feedback_qas is True and feedback.stages() == ["feedback_weights"]

    distill = await probe_session_pending_work(
        FakeSessionManager(
            qa=[_qa()],
            trace_count=4,
            context=[
                *base_context,
                _entry("2026-09-09T11:00:00+00:00"),
                _distilled("2026-09-09T10:00:00+00:00"),
            ],
        ),
        user_id="u",
        session_id="s",
    )
    assert distill.distillable_entries is True and distill.stages() == ["distill_sessions"]


@pytest.mark.asyncio
async def test_low_confidence_or_harmful_guidance_is_not_distillable():
    manager = FakeSessionManager(
        context=[_entry(confidence=0.2), _entry("2026-09-09T12:00:00+00:00", harmful=1)]
    )

    work = await probe_session_pending_work(manager, user_id="u", session_id="s")

    assert work.distillable_entries is False


@pytest.mark.asyncio
async def test_agent_context_stage_respects_auto_feedback_flag():
    manager = FakeSessionManager(trace_count=3, auto_feedback=False, context=[_trace_persist(3)])

    work = await probe_session_pending_work(manager, user_id="u", session_id="s")

    assert work.new_traces_for_agent_context is False
    assert work.new_traces_to_persist is False


@pytest.mark.asyncio
async def test_stale_watermarks_count_as_pending():
    """A watermark above the current count means a rebuilt session: re-bridge it."""
    manager = FakeSessionManager(
        qa=[_qa()], trace_count=1, context=[_persist(9), _trace_persist(9)]
    )

    work = await probe_session_pending_work(manager, user_id="u", session_id="s")

    assert work.new_qa is True
    assert work.new_traces_to_persist is True


@pytest.mark.asyncio
async def test_cache_failure_reports_everything_pending():
    """Fail towards doing the (watermark-guarded) work rather than skipping a bridge."""
    work = await probe_session_pending_work(
        FakeSessionManager(fail=True), user_id="u", session_id="s"
    )

    assert work.any is True
    assert set(work.stages()) == {
        "feedback_weights",
        "persist_sessions",
        "persist_trace_steps",
        "extract_agent_context",
        "distill_sessions",
    }


@pytest.mark.asyncio
async def test_probe_sessions_keys_by_session_id():
    pending = await probe_sessions_pending_work(
        FakeSessionManager(qa=[_qa()]), user_id="u", session_ids=["a", "b"]
    )

    assert set(pending) == {"a", "b"}
    assert all(work.new_qa for work in pending.values())
