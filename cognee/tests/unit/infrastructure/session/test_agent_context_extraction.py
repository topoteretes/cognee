"""Unit tests for agent-context extraction (live deterministic pass + LLM batch pass).

Uses fake session managers and a fake LLM so the tests run anywhere; verifies that only errored
steps produce a live failure_lessons lesson, that the batch pass stores LLM-proposed lessons, and
the fail-open contract for both.
"""

from types import SimpleNamespace

import pytest

from cognee.infrastructure.session import agent_context_extraction
from cognee.infrastructure.session.agent_context_extraction import (
    LIVE_FAILURE_CONFIDENCE,
    MAX_BATCH_LESSONS,
    build_live_agent_candidates,
    build_trace_batch,
    extract_batch_agent_context,
    extract_live_agent_context,
)
from cognee.infrastructure.session.session_context_models import (
    AgentContextExtraction,
    SessionContextEntry,
)


def _trace(
    origin_function, status="success", session_feedback="", error_message="", return_value=None
):
    return SimpleNamespace(
        origin_function=origin_function,
        status=status,
        session_feedback=session_feedback,
        error_message=error_message,
        method_return_value=return_value,
    )


class FakeSessionManager:
    """In-memory stand-in for the context CRUD + trace-read surface."""

    def __init__(self, traces=None):
        self.store = []
        self.traces = list(traces or [])

    async def get_session_context_entries(self, user_id, session_id):
        return list(self.store)

    async def create_session_context_entry(self, user_id, session_id, entry_dump):
        self.store.append(entry_dump)

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        for row in self.store:
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def get_agent_trace_session(self, user_id, session_id, last_n=None):
        return list(self.traces)


class RaisingSessionManager:
    async def get_session_context_entries(self, user_id, session_id):
        raise RuntimeError("boom")

    async def create_session_context_entry(self, user_id, session_id, entry_dump):
        raise RuntimeError("boom")

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        raise RuntimeError("boom")


def test_errored_step_builds_failure_candidate():
    candidates = build_live_agent_candidates(
        origin_function="run_tests", status="error", error_message="ModuleNotFoundError: dotenv"
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.section == "failure_lessons"
    assert candidate.context_profile == "agent"
    assert candidate.confidence == LIVE_FAILURE_CONFIDENCE
    assert "run_tests failed" in candidate.content
    assert "dotenv" in candidate.content


def test_successful_step_builds_no_candidate():
    assert (
        build_live_agent_candidates(origin_function="run_tests", status="success", error_message="")
        == []
    )


def test_errored_step_without_message_builds_no_candidate():
    assert (
        build_live_agent_candidates(
            origin_function="run_tests", status="error", error_message="   "
        )
        == []
    )


@pytest.mark.asyncio
async def test_extract_live_stores_agent_entry_linked_to_trace():
    sm = FakeSessionManager()
    touched = await extract_live_agent_context(
        session_manager=sm,
        user_id="u",
        session_id="s",
        trace_id="trace-1",
        origin_function="run_tests",
        status="error",
        error_message="exit code 1",
    )
    assert len(touched) == 1
    created = sm.store[0]
    assert created["context_profile"] == "agent"
    assert created["section"] == "failure_lessons"
    assert created["source_trace_ids"] == ["trace-1"]
    assert created["source_feedback_ids"] == []


@pytest.mark.asyncio
async def test_extract_live_noop_on_success():
    sm = FakeSessionManager()
    touched = await extract_live_agent_context(
        session_manager=sm,
        user_id="u",
        session_id="s",
        trace_id="trace-1",
        origin_function="run_tests",
        status="success",
        error_message="",
    )
    assert touched == []
    assert sm.store == []


@pytest.mark.asyncio
async def test_extract_live_fail_open_on_raising_manager():
    sm = RaisingSessionManager()
    touched = await extract_live_agent_context(
        session_manager=sm,
        user_id="u",
        session_id="s",
        trace_id="trace-1",
        origin_function="run_tests",
        status="error",
        error_message="exit code 1",
    )
    assert touched == []


# ------------------------------------------------------------------- batch (LLM) pass


def _patch_llm(monkeypatch, lessons, captured=None):
    async def fake_acreate_structured_output(text_input, system_prompt, response_model):
        if captured is not None:
            captured["text_input"] = text_input
        return AgentContextExtraction(lessons=lessons)

    monkeypatch.setattr(
        agent_context_extraction.LLMGateway,
        "acreate_structured_output",
        fake_acreate_structured_output,
    )


def test_build_trace_batch_renders_compact_lines():
    traces = [
        _trace("run_tests", status="error", error_message="exit 1"),
        _trace(
            "read_file", status="success", session_feedback="read config", return_value={"k": "v"}
        ),
    ]
    batch = build_trace_batch(traces)
    assert "run_tests [error]" in batch
    assert "error: exit 1" in batch
    assert "read_file [success]" in batch
    assert "output:" in batch


@pytest.mark.asyncio
async def test_batch_stores_llm_lessons(monkeypatch):
    sm = FakeSessionManager(traces=[_trace("run_tests", status="error", error_message="exit 1")])
    _patch_llm(
        monkeypatch,
        lessons=[
            {
                "section": "environment_facts",
                "content": "Tests need uv sync first",
                "confidence": 0.9,
            },
            {"section": "tool_rules", "content": "Run tests with uv run pytest", "confidence": 0.8},
        ],
    )
    touched = await extract_batch_agent_context(session_manager=sm, user_id="u", session_id="s")
    assert len(touched) == 2
    sections = {row["section"] for row in sm.store}
    assert sections == {"environment_facts", "tool_rules"}
    assert all(row["context_profile"] == "agent" for row in sm.store)
    # Batch lessons have no single source trace.
    assert all(row["source_trace_ids"] == [] for row in sm.store)


@pytest.mark.asyncio
async def test_batch_noop_without_traces(monkeypatch):
    sm = FakeSessionManager(traces=[])
    _patch_llm(monkeypatch, lessons=[{"section": "tool_rules", "content": "x", "confidence": 0.9}])
    touched = await extract_batch_agent_context(session_manager=sm, user_id="u", session_id="s")
    assert touched == []
    assert sm.store == []


@pytest.mark.asyncio
async def test_batch_fail_open_on_llm_error(monkeypatch):
    sm = FakeSessionManager(traces=[_trace("run_tests", status="error", error_message="exit 1")])

    async def boom(text_input, system_prompt, response_model):
        raise RuntimeError("llm down")

    monkeypatch.setattr(agent_context_extraction.LLMGateway, "acreate_structured_output", boom)
    touched = await extract_batch_agent_context(session_manager=sm, user_id="u", session_id="s")
    assert touched == []
    assert sm.store == []


@pytest.mark.asyncio
async def test_batch_shows_existing_lessons_to_the_model(monkeypatch):
    sm = FakeSessionManager(traces=[_trace("run_tests", status="error", error_message="exit 1")])
    sm.store.append(
        SessionContextEntry(
            id="e1",
            section="failure_lessons",
            context_profile="agent",
            content="Old lesson about uv.",
            created_at="2026-06-11T10:00:00",
        ).model_dump()
    )
    captured = {}
    _patch_llm(monkeypatch, lessons=[], captured=captured)

    await extract_batch_agent_context(session_manager=sm, user_id="u", session_id="s")

    assert "EXISTING LESSONS" in captured["text_input"]
    assert "Old lesson about uv." in captured["text_input"]
    assert "TRACES:" in captured["text_input"]


@pytest.mark.asyncio
async def test_batch_caps_new_lessons_per_run(monkeypatch):
    sm = FakeSessionManager(traces=[_trace("run_tests", status="error", error_message="exit 1")])
    # Seven distinct, novel lessons; only MAX_BATCH_LESSONS should be applied.
    lessons = [
        {"section": "tool_rules", "content": f"Lesson number {i}.", "confidence": 0.9}
        for i in range(7)
    ]
    _patch_llm(monkeypatch, lessons=lessons)

    touched = await extract_batch_agent_context(session_manager=sm, user_id="u", session_id="s")

    assert len(touched) == MAX_BATCH_LESSONS
    assert len(sm.store) == MAX_BATCH_LESSONS
