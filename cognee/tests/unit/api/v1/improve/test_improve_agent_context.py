"""Unit tests for the improve() agent-context extraction stage (stage 4).

The stage iterates sessions, is gated on automatic session context, and is fail-open per
session. These tests stub the session manager and pending extractor so no LLM/cache is needed.
"""

import importlib
import types
from uuid import uuid4

import pytest

from cognee.modules.improve import GraphCapabilities, ImproveConfig, ImproveRunInputs
from cognee.modules.improve.stages import (
    REASON_AUTO_FEEDBACK_DISABLED,
    REASON_SESSION_MANAGER_UNAVAILABLE,
    ExtractAgentContextStage,
)


def _inputs(session_ids):
    user = types.SimpleNamespace(id=uuid4())
    dataset = types.SimpleNamespace(id=uuid4(), name="docs", owner_id=user.id)
    return ImproveRunInputs(
        user=user,
        dataset_id=dataset.id,
        dataset=dataset,
        session_ids=tuple(session_ids),
        config=ImproveConfig(),
        capabilities=GraphCapabilities.assume_supported(),
    )


def _patch_session_manager(monkeypatch, *, available=True, auto_feedback=True):
    fake_sm = types.SimpleNamespace(
        is_available=available,
        is_auto_feedback_enabled=lambda: auto_feedback,
    )
    gsm_mod = importlib.import_module("cognee.infrastructure.session.get_session_manager")
    monkeypatch.setattr(gsm_mod, "get_session_manager", lambda: fake_sm)
    return fake_sm


def _patch_extractor(monkeypatch, behavior):
    ace_mod = importlib.import_module("cognee.infrastructure.session.agent_context_extraction")
    calls = []

    async def fake_extract(*, session_manager, user_id, session_id, min_new_traces):
        calls.append((session_id, min_new_traces))
        return behavior(session_id)

    monkeypatch.setattr(ace_mod, "extract_pending_agent_context", fake_extract)
    return calls


@pytest.mark.asyncio
async def test_runs_extraction_per_session_and_counts_lessons(monkeypatch):
    _patch_session_manager(monkeypatch)
    calls = _patch_extractor(monkeypatch, behavior=lambda _sid: ["lesson"])
    stage = ExtractAgentContextStage()
    inputs = _inputs(["s1", "s2"])

    assert stage.gate(inputs) is None
    result = await stage.run(inputs)

    assert calls == [("s1", 1), ("s2", 1)]
    assert result.status == "completed"
    assert result.counts == {"lessons": 2, "sessions_failed": 0}


def test_gate_skips_when_auto_feedback_disabled(monkeypatch):
    _patch_session_manager(monkeypatch, auto_feedback=False)
    assert ExtractAgentContextStage().gate(_inputs(["s1"])) == REASON_AUTO_FEEDBACK_DISABLED


def test_gate_skips_when_session_manager_unavailable(monkeypatch):
    _patch_session_manager(monkeypatch, available=False)
    assert ExtractAgentContextStage().gate(_inputs(["s1"])) == REASON_SESSION_MANAGER_UNAVAILABLE


@pytest.mark.asyncio
async def test_one_failing_session_does_not_block_others(monkeypatch):
    _patch_session_manager(monkeypatch)

    def behavior(session_id):
        if session_id == "s1":
            raise RuntimeError("boom")
        return ["lesson"]

    calls = _patch_extractor(monkeypatch, behavior=behavior)

    result = await ExtractAgentContextStage().run(_inputs(["s1", "s2"]))

    assert calls == [("s1", 1), ("s2", 1)]  # s2 still processed after s1 raised
    assert result.status == "errored"
    assert result.counts == {"lessons": 1, "sessions_failed": 1}
    assert "boom" in result.error
