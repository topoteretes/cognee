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
    TRACE_EXTRACTION_STATE_ID,
    TRACE_EXTRACTION_STATE_KIND,
    build_live_agent_candidates,
    build_trace_batch,
    extract_batch_agent_context,
    extract_live_agent_context,
    extract_pending_agent_context,
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
        self.trace_session_last_n_calls = []

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
        self.trace_session_last_n_calls.append(last_n)
        if last_n is not None:
            return list(self.traces[-last_n:])
        return list(self.traces)

    async def get_agent_trace_count(self, user_id, session_id):
        return len(self.traces)


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


def test_errored_step_redacts_sensitive_error_text():
    candidates = build_live_agent_candidates(
        origin_function="call_api",
        status="error",
        error_message=(
            "Request failed: Bearer sk-live-secret token=abc123 "
            "request_id=123456789 path /tmp/550e8400-e29b-41d4-a716-446655440000"
        ),
    )

    assert len(candidates) == 1
    content = candidates[0].content
    assert "sk-live-secret" not in content
    assert "abc123" not in content
    assert "123456789" not in content
    assert "550e8400-e29b-41d4-a716-446655440000" not in content
    assert "Bearer [redacted]" in content
    assert "token=[redacted]" in content
    assert "request_id=[number]" in content
    assert "[uuid]" in content


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


def test_build_trace_batch_redacts_sensitive_error_text():
    batch = build_trace_batch(
        [
            _trace(
                "fetch_remote",
                status="error",
                error_message=(
                    "HTTP 401 api_key=super-secret jwt=eyJabc.def.ghi request 987654321"
                ),
            )
        ]
    )

    assert "super-secret" not in batch
    assert "eyJabc.def.ghi" not in batch
    assert "987654321" not in batch
    assert "api_key=[redacted]" in batch
    assert "jwt=[redacted]" in batch
    assert "request [number]" in batch


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


# --------------------------------------------------------------- pending pass


def _state_row(processed_trace_count):
    return {
        "id": TRACE_EXTRACTION_STATE_ID,
        "kind": TRACE_EXTRACTION_STATE_KIND,
        "processed_trace_count": processed_trace_count,
    }


@pytest.mark.asyncio
async def test_pending_extraction_noop_below_interval(monkeypatch):
    sm = FakeSessionManager(traces=[_trace("step-1"), _trace("step-2")])

    async def unexpected_llm(text_input, system_prompt, response_model):
        raise AssertionError("LLM should not run below interval")

    monkeypatch.setattr(
        agent_context_extraction.LLMGateway,
        "acreate_structured_output",
        unexpected_llm,
    )

    touched = await extract_pending_agent_context(
        session_manager=sm, user_id="u", session_id="s", min_new_traces=3, overlap=1
    )

    assert touched == []
    assert sm.store == []


@pytest.mark.asyncio
async def test_pending_extraction_runs_at_interval_and_sets_watermark(monkeypatch):
    sm = FakeSessionManager(traces=[_trace("step-1"), _trace("step-2"), _trace("step-3")])
    _patch_llm(
        monkeypatch,
        lessons=[
            {
                "section": "success_patterns",
                "content": "Batch trace windows can produce reusable lessons.",
                "confidence": 0.9,
            }
        ],
    )

    touched = await extract_pending_agent_context(
        session_manager=sm, user_id="u", session_id="s", min_new_traces=3, overlap=1
    )

    assert len(touched) == 1
    assert sm.trace_session_last_n_calls == [4]
    state = next(row for row in sm.store if row.get("kind") == TRACE_EXTRACTION_STATE_KIND)
    assert state["processed_trace_count"] == 3


@pytest.mark.asyncio
async def test_pending_extraction_can_flush_below_interval(monkeypatch):
    sm = FakeSessionManager(traces=[_trace("step-1"), _trace("step-2")])
    _patch_llm(
        monkeypatch,
        lessons=[
            {
                "section": "tool_rules",
                "content": "Flush pending traces before distillation.",
                "confidence": 0.9,
            }
        ],
    )

    touched = await extract_pending_agent_context(
        session_manager=sm, user_id="u", session_id="s", min_new_traces=1, overlap=1
    )

    assert len(touched) == 1
    assert sm.trace_session_last_n_calls == [3]
    state = next(row for row in sm.store if row.get("kind") == TRACE_EXTRACTION_STATE_KIND)
    assert state["processed_trace_count"] == 2


@pytest.mark.asyncio
async def test_pending_extraction_uses_overlap_after_watermark(monkeypatch):
    sm = FakeSessionManager(traces=[_trace(f"step-{index}") for index in range(13)])
    sm.store.append(_state_row(10))
    captured = {}
    _patch_llm(monkeypatch, lessons=[], captured=captured)

    touched = await extract_pending_agent_context(
        session_manager=sm, user_id="u", session_id="s", min_new_traces=3, overlap=2
    )

    assert touched == []
    assert sm.trace_session_last_n_calls == [5]
    assert "step-8" in captured["text_input"]
    assert "step-12" in captured["text_input"]
    state = next(row for row in sm.store if row.get("kind") == TRACE_EXTRACTION_STATE_KIND)
    assert state["processed_trace_count"] == 13


@pytest.mark.asyncio
async def test_pending_extraction_does_not_advance_watermark_on_llm_error(monkeypatch):
    sm = FakeSessionManager(traces=[_trace("step-1"), _trace("step-2"), _trace("step-3")])

    async def boom(text_input, system_prompt, response_model):
        raise RuntimeError("llm down")

    monkeypatch.setattr(agent_context_extraction.LLMGateway, "acreate_structured_output", boom)

    touched = await extract_pending_agent_context(
        session_manager=sm, user_id="u", session_id="s", min_new_traces=3, overlap=1
    )

    assert touched == []
    assert not any(row.get("kind") == TRACE_EXTRACTION_STATE_KIND for row in sm.store)
