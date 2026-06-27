"""Unit tests for the improve() agent-context extraction stage (_extract_agent_context).

The stage iterates sessions, is gated on automatic session context, and is fail-open per
session. These tests stub the session manager and pending extractor so no LLM/cache is needed.
"""

import importlib
import types
from uuid import uuid4

import pytest


def _make_user():
    return types.SimpleNamespace(id=uuid4())


@pytest.fixture
def improve_mod():
    return importlib.import_module("cognee.api.v1.improve.improve")


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
async def test_runs_extraction_per_session_and_counts_lessons(monkeypatch, improve_mod):
    _patch_session_manager(monkeypatch)
    calls = _patch_extractor(monkeypatch, behavior=lambda _sid: ["lesson"])

    total = await improve_mod._extract_agent_context(session_ids=["s1", "s2"], user=_make_user())

    assert calls == [("s1", 1), ("s2", 1)]
    assert total == 2


@pytest.mark.asyncio
async def test_skipped_when_auto_feedback_disabled(monkeypatch, improve_mod):
    _patch_session_manager(monkeypatch, auto_feedback=False)
    calls = _patch_extractor(monkeypatch, behavior=lambda _sid: ["lesson"])

    total = await improve_mod._extract_agent_context(session_ids=["s1"], user=_make_user())

    assert total == 0
    assert calls == []


@pytest.mark.asyncio
async def test_one_failing_session_does_not_block_others(monkeypatch, improve_mod):
    _patch_session_manager(monkeypatch)

    def behavior(session_id):
        if session_id == "s1":
            raise RuntimeError("boom")
        return ["lesson"]

    calls = _patch_extractor(monkeypatch, behavior=behavior)

    total = await improve_mod._extract_agent_context(session_ids=["s1", "s2"], user=_make_user())

    assert calls == [("s1", 1), ("s2", 1)]  # s2 still processed after s1 raised
    assert total == 1
