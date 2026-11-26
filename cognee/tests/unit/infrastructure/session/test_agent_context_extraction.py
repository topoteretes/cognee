"""Unit tests for agent-context extraction (live deterministic pass + LLM batch pass).

Uses fake session managers and a fake LLM so the tests run anywhere; verifies that only errored
steps produce a live failure_lessons lesson, that the batch pass stores LLM-proposed lessons, and
the fail-open contract for both.
"""

import asyncio
from types import SimpleNamespace

import pytest

from cognee.infrastructure.session import agent_context_extraction
from cognee.infrastructure.session.agent_context_extraction import (
    LIVE_FAILURE_CONFIDENCE,
    MAX_BATCH_LESSONS,
    TRACE_EXTRACTION_STATE_ID,
    TRACE_EXTRACTION_STATE_KIND,
    TRACE_EXTRACTION_WATERMARK,
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

    dataset_id = None

    def __init__(self, traces=None):
        self.store = []
        self.traces = list(traces or [])
        self.trace_session_last_n_calls = []
        self.trace_ranges = []
        self.traces_read = 0

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

    async def get_agent_trace_session(self, user_id, session_id, last_n=None, start=None, end=None):
        self.trace_session_last_n_calls.append(last_n)
        if last_n is not None:
            traces = list(self.traces[-last_n:])
        elif start is not None:
            self.trace_ranges.append((start, end))
            traces = list(self.traces[start:end])
        else:
            traces = list(self.traces)
        self.traces_read += len(traces)
        return traces

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
        return AgentContextExtraction(
            lessons=[
                {"evidence": "The test trace demonstrates the repository constraint.", **lesson}
                for lesson in lessons
            ]
        )

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
async def test_batch_llm_error_propagates(monkeypatch):
    """An LLM outage is not "no lessons found": it must reach the caller so the
    watermark never advances over traces that were never extracted, and so the
    improve stage's errored branch is reachable."""
    sm = FakeSessionManager(traces=[_trace("run_tests", status="error", error_message="exit 1")])

    async def boom(text_input, system_prompt, response_model):
        raise RuntimeError("llm down")

    monkeypatch.setattr(agent_context_extraction.LLMGateway, "acreate_structured_output", boom)
    with pytest.raises(RuntimeError, match="llm down"):
        await extract_batch_agent_context(session_manager=sm, user_id="u", session_id="s")
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
async def test_pending_extraction_noop_without_a_usable_llm(monkeypatch):
    sm = FakeSessionManager(traces=[_trace("step-1"), _trace("step-2"), _trace("step-3")])

    async def unexpected_llm(text_input, system_prompt, response_model):
        raise AssertionError("LLM should not run when none is configured")

    monkeypatch.setattr(agent_context_extraction, "llm_available", lambda: False)
    monkeypatch.setattr(
        agent_context_extraction.LLMGateway,
        "acreate_structured_output",
        unexpected_llm,
    )

    touched = await extract_pending_agent_context(
        session_manager=sm, user_id="u", session_id="s", min_new_traces=3, overlap=1
    )

    assert touched == []
    assert sm.trace_session_last_n_calls == []
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
    assert "step-7 " not in captured["text_input"]
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

    with pytest.raises(RuntimeError, match="llm down"):
        await extract_pending_agent_context(
            session_manager=sm, user_id="u", session_id="s", min_new_traces=3, overlap=1
        )

    state = next(row for row in sm.store if row.get("kind") == TRACE_EXTRACTION_STATE_KIND)
    assert "processed_trace_count" not in state
    assert state["attempted_trace_count"] == 3


USER_ID = "6f1c2c9e-4b8a-4a51-9a35-5d0a1f3e7b21"


async def _write_traces(sm, count, admission=None):
    """Store ``count`` traces one at a time, each followed by the trace-write path's
    extraction. Its failures are returned instead of raised, as SessionManager logs
    them instead of failing the write."""
    failures = []
    for _ in range(count):
        sm.traces.append(_trace(f"step-{len(sm.traces)}"))
        try:
            await extract_pending_agent_context(
                session_manager=sm, user_id=USER_ID, session_id="s", admission=admission
            )
        except RuntimeError as error:
            failures.append(error)
    return failures


def _count_llm_calls(monkeypatch, error=None, lessons=()):
    calls = []

    async def llm(text_input, system_prompt, response_model):
        calls.append(text_input)
        if error is not None:
            raise error
        return AgentContextExtraction(lessons=list(lessons))

    monkeypatch.setattr(agent_context_extraction, "llm_available", lambda: True)
    monkeypatch.setattr(agent_context_extraction.LLMGateway, "acreate_structured_output", llm)
    return calls


@pytest.mark.asyncio
async def test_a_call_that_always_fails_is_made_once_per_interval(monkeypatch):
    # A spent budget, a bad key or a missing model fails every call. Counting from the
    # last finished window alone made one call per trace write after the tenth.
    calls = _count_llm_calls(monkeypatch, error=RuntimeError("Budget has been exceeded"))
    sm = FakeSessionManager()

    failures = await _write_traces(sm, 30)

    assert len(calls) == 3  # at traces 10, 20 and 30
    assert len(failures) == 3


@pytest.mark.asyncio
async def test_a_cancelled_call_waits_for_the_interval_too(monkeypatch):
    # The start is recorded before the call, so a call that never returns (a cancelled
    # task, a killed worker) is not retried on the next trace write either.
    calls = _count_llm_calls(monkeypatch, error=asyncio.CancelledError())
    sm = FakeSessionManager()

    with pytest.raises(asyncio.CancelledError):
        await _write_traces(sm, 10)
    await _write_traces(sm, 9)

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_the_window_after_a_failed_one_covers_its_traces(monkeypatch):
    calls = _count_llm_calls(monkeypatch, error=RuntimeError("llm down"))
    sm = FakeSessionManager()
    await _write_traces(sm, 10)
    calls = _count_llm_calls(monkeypatch)

    await _write_traces(sm, 10)

    assert len(calls) == 1
    assert "step-0 " in calls[0] and "step-19 " in calls[0]
    assert await TRACE_EXTRACTION_WATERMARK.read_count(sm, USER_ID, "s") == 20


@pytest.mark.asyncio
async def test_improve_retries_an_unfinished_window_at_once(monkeypatch):
    calls = _count_llm_calls(monkeypatch, error=RuntimeError("llm down"))
    sm = FakeSessionManager()
    await _write_traces(sm, 10)
    calls = _count_llm_calls(monkeypatch)

    await extract_pending_agent_context(
        session_manager=sm, user_id=USER_ID, session_id="s", min_new_traces=1, automatic=False
    )

    assert len(calls) == 1
    assert await TRACE_EXTRACTION_WATERMARK.read_count(sm, USER_ID, "s") == 10


@pytest.mark.asyncio
async def test_a_finished_window_keeps_the_attempt_and_the_watermark(monkeypatch):
    _count_llm_calls(monkeypatch)
    sm = FakeSessionManager()

    await _write_traces(sm, 10)

    state = next(row for row in sm.store if row.get("kind") == TRACE_EXTRACTION_STATE_KIND)
    assert state["attempted_trace_count"] == 10
    assert state["processed_trace_count"] == 10


@pytest.mark.asyncio
async def test_traces_gone_below_the_recorded_attempt_start_over(monkeypatch):
    # The trace key can expire while the context row lives on (Redis TTL). An attempt
    # count above the trace count is stale, like the processed count above it.
    calls = _count_llm_calls(monkeypatch, error=RuntimeError("llm down"))
    sm = FakeSessionManager()
    await _write_traces(sm, 10)
    sm.traces.clear()
    calls = _count_llm_calls(monkeypatch)

    await _write_traces(sm, 10)

    assert len(calls) == 1


class YieldingSessionManager(FakeSessionManager):
    """Yields to the event loop on every store call, as a real cache backend does."""

    async def get_session_context_entries(self, user_id, session_id):
        await asyncio.sleep(0)
        return await super().get_session_context_entries(user_id, session_id)

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        await asyncio.sleep(0)
        return await super().update_session_context_entry(user_id, session_id, entry_id, merge)

    async def get_agent_trace_count(self, user_id, session_id):
        await asyncio.sleep(0)
        return await super().get_agent_trace_count(user_id, session_id)


@pytest.mark.asyncio
async def test_concurrent_trace_writes_start_one_window(monkeypatch):
    calls = _count_llm_calls(monkeypatch, error=RuntimeError("llm down"))
    sm = YieldingSessionManager(traces=[_trace(f"step-{i}") for i in range(10)])

    results = await asyncio.gather(
        *(
            extract_pending_agent_context(session_manager=sm, user_id=USER_ID, session_id="s")
            for _ in range(5)
        ),
        return_exceptions=True,
    )

    assert len(calls) == 1
    assert sum(isinstance(result, RuntimeError) for result in results) == 1


@pytest.mark.asyncio
async def test_a_slow_window_does_not_move_the_watermark_back_under_improve(monkeypatch):
    # The trace path's window is still in its LLM call when improve() flushes the
    # session. Improve waits for it, so its own write is the last one.
    release = asyncio.Event()
    calls = []

    async def llm(text_input, system_prompt, response_model):
        calls.append(text_input)
        if len(calls) == 1:
            await release.wait()
        return AgentContextExtraction(lessons=[])

    monkeypatch.setattr(agent_context_extraction, "llm_available", lambda: True)
    monkeypatch.setattr(agent_context_extraction.LLMGateway, "acreate_structured_output", llm)
    sm = YieldingSessionManager(traces=[_trace(f"step-{i}") for i in range(10)])

    automatic = asyncio.create_task(
        extract_pending_agent_context(session_manager=sm, user_id=USER_ID, session_id="s")
    )
    while not calls:
        await asyncio.sleep(0)
    sm.traces.extend(_trace(f"step-{i}") for i in range(10, 20))
    flush = asyncio.create_task(
        extract_pending_agent_context(
            session_manager=sm, user_id=USER_ID, session_id="s", min_new_traces=1, automatic=False
        )
    )
    for _ in range(50):
        await asyncio.sleep(0)
    release.set()
    await asyncio.gather(automatic, flush)

    assert await TRACE_EXTRACTION_WATERMARK.read_count(sm, USER_ID, "s") == 20


@pytest.mark.asyncio
async def test_a_failed_improve_run_delays_the_trace_path(monkeypatch):
    calls = _count_llm_calls(monkeypatch, error=RuntimeError("llm down"))
    sm = FakeSessionManager(traces=[_trace(f"step-{i}") for i in range(9)])
    with pytest.raises(RuntimeError):
        await extract_pending_agent_context(
            session_manager=sm, user_id=USER_ID, session_id="s", min_new_traces=1, automatic=False
        )

    await _write_traces(sm, 9)

    assert len(calls) == 1


def _admission(answer=None):
    asked = []

    async def admission():
        asked.append(True)
        return answer

    return admission, asked


@pytest.mark.asyncio
async def test_the_admission_skips_the_call_once_per_interval(monkeypatch):
    admission, asked = _admission("insufficient_credits")
    calls = _count_llm_calls(monkeypatch)
    sm = FakeSessionManager()

    await _write_traces(sm, 20, admission=admission)

    assert calls == []
    assert len(asked) == 2  # at traces 10 and 20, not on every write
    assert await TRACE_EXTRACTION_WATERMARK.read_count(sm, USER_ID, "s") == 0


@pytest.mark.asyncio
async def test_a_hung_admission_does_not_hold_the_session_lock(monkeypatch):
    # improve() holds the dataset lock while its stage 4 takes this session's lock,
    # so a host check that never answers must not be asked under that lock.
    calls = _count_llm_calls(monkeypatch)
    asked = asyncio.Event()

    async def hung_admission():
        asked.set()
        await asyncio.Event().wait()

    sm = FakeSessionManager(traces=[_trace(f"step-{i}") for i in range(10)])
    trace_path = asyncio.create_task(
        extract_pending_agent_context(
            session_manager=sm, user_id=USER_ID, session_id="s", admission=hung_admission
        )
    )
    try:
        await asyncio.wait_for(asked.wait(), timeout=1)
        await asyncio.wait_for(
            extract_pending_agent_context(
                session_manager=sm,
                user_id=USER_ID,
                session_id="s",
                min_new_traces=1,
                automatic=False,
            ),
            timeout=1,
        )
    finally:
        # The session lock lives in a process-wide registry: a task left holding it
        # would hang every later test on this session.
        trace_path.cancel()
        await asyncio.gather(trace_path, return_exceptions=True)

    assert len(calls) == 1
    assert await TRACE_EXTRACTION_WATERMARK.read_count(sm, USER_ID, "s") == 10


@pytest.mark.asyncio
async def test_a_backlog_above_the_window_is_extracted_oldest_first(monkeypatch):
    # Four failed windows leave 50 traces pending, more than one window holds.
    _count_llm_calls(monkeypatch, error=RuntimeError("llm down"))
    sm = FakeSessionManager()
    await _write_traces(sm, 49)
    calls = _count_llm_calls(monkeypatch)

    await _write_traces(sm, 1)

    sent = "".join(calls)
    assert all(f"step-{i} " in sent for i in range(50))
    assert "step-0 " in calls[0]
    assert await TRACE_EXTRACTION_WATERMARK.read_count(sm, USER_ID, "s") == 50


@pytest.mark.asyncio
async def test_a_window_reads_only_its_own_traces(monkeypatch):
    # A long-lived session must not be read whole on every due window.
    calls = _count_llm_calls(monkeypatch)
    sm = FakeSessionManager(traces=[_trace(f"step-{i}") for i in range(2000)])
    await TRACE_EXTRACTION_WATERMARK.write_count(sm, USER_ID, "s", count=1990)

    await extract_pending_agent_context(
        session_manager=sm, user_id=USER_ID, session_id="s", automatic=False
    )

    assert sm.traces_read == 13  # 10 new traces and the overlap of 3
    assert "step-1987 " in calls[0] and "step-1986 " not in calls[0]


@pytest.mark.asyncio
async def test_traces_written_after_the_plan_wait_for_the_next_window(monkeypatch):
    # Trace writes take no lock, so traces can land between the plan and the read.
    # The window must still cover exactly the planned traces, oldest first.
    calls = _count_llm_calls(monkeypatch)
    sm = FakeSessionManager(traces=[_trace(f"step-{i}") for i in range(20)])
    await TRACE_EXTRACTION_WATERMARK.write_count(sm, USER_ID, "s", count=10)
    read_traces = sm.get_agent_trace_session

    async def append_then_read(**kwargs):
        sm.traces.extend(_trace(f"late-{i}") for i in range(3))
        return await read_traces(**kwargs)

    sm.get_agent_trace_session = append_then_read

    await extract_pending_agent_context(
        session_manager=sm, user_id=USER_ID, session_id="s", automatic=False
    )

    assert all(f"step-{i} " in calls[0] for i in range(10, 20))
    assert "late-" not in calls[0]
    assert sm.trace_ranges == [(7, 20)]  # the read stops at the planned total
    assert await TRACE_EXTRACTION_WATERMARK.read_count(sm, USER_ID, "s") == 20


@pytest.mark.asyncio
async def test_an_admission_that_writes_a_trace_to_the_session_finishes(monkeypatch):
    # A host check that records its own decision as a trace in the same session
    # runs that trace's extraction; it must not wait on a lock its caller holds.
    calls = _count_llm_calls(monkeypatch)
    sm = FakeSessionManager(traces=[_trace(f"step-{i}") for i in range(10)])

    async def admission():
        sm.traces.append(_trace("admission-decision"))
        await extract_pending_agent_context(
            session_manager=sm, user_id=USER_ID, session_id="s", min_new_traces=1, automatic=False
        )

    await asyncio.wait_for(
        extract_pending_agent_context(
            session_manager=sm, user_id=USER_ID, session_id="s", admission=admission
        ),
        timeout=1,
    )

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_an_allowing_admission_lets_the_call_run(monkeypatch):
    admission, asked = _admission(None)
    calls = _count_llm_calls(monkeypatch)
    sm = FakeSessionManager()

    await _write_traces(sm, 10, admission=admission)

    assert len(asked) == 1
    assert len(calls) == 1


# ------------------------------------------------------- shared watermark implementation


def test_stage4_watermark_is_the_shared_state_row_helper():
    from cognee.infrastructure.session.agent_context_extraction import TRACE_EXTRACTION_WATERMARK
    from cognee.infrastructure.session.session_persist_watermark import StateRowWatermark

    assert isinstance(TRACE_EXTRACTION_WATERMARK, StateRowWatermark)
    assert TRACE_EXTRACTION_WATERMARK.state_id == TRACE_EXTRACTION_STATE_ID
    assert TRACE_EXTRACTION_WATERMARK.state_kind == TRACE_EXTRACTION_STATE_KIND
    # Rows written before the shared helper existed keep being read.
    assert TRACE_EXTRACTION_WATERMARK.count_from_rows([_state_row(10)]) == 10


@pytest.mark.asyncio
async def test_pending_extraction_reads_and_writes_through_the_shared_helper(monkeypatch):
    from cognee.infrastructure.session.agent_context_extraction import TRACE_EXTRACTION_WATERMARK

    sm = FakeSessionManager(traces=[_trace(f"step-{index}") for index in range(5)])
    sm.store.append(_state_row(3))
    captured = {}
    _patch_llm(monkeypatch, lessons=[], captured=captured)

    await extract_pending_agent_context(
        session_manager=sm, user_id="u", session_id="s", min_new_traces=1, overlap=0
    )

    # 5 total - 3 processed = 2 pending, sent with no overlap.
    assert "step-2 " not in captured["text_input"]
    assert "step-3 " in captured["text_input"] and "step-4 " in captured["text_input"]
    assert await TRACE_EXTRACTION_WATERMARK.read_count(sm, "u", "s") == 5
    # Still exactly one state row, updated in place.
    assert len([row for row in sm.store if row.get("kind") == TRACE_EXTRACTION_STATE_KIND]) == 1


@pytest.mark.asyncio
async def test_concurrent_pending_extractions_send_the_window_once(monkeypatch):
    # The trace write's background batch and improve's extract_agent_context stage
    # can run at once on one session; the second must see the advanced watermark.
    import asyncio

    count = agent_context_extraction.TRACE_EXTRACTION_INTERVAL
    sm = FakeSessionManager(traces=[_trace(f"step-{i}") for i in range(count)])
    calls = 0
    release = asyncio.Event()

    async def slow_llm(text_input, system_prompt, response_model):
        nonlocal calls
        calls += 1
        await release.wait()
        return AgentContextExtraction(lessons=[])

    monkeypatch.setattr(agent_context_extraction.LLMGateway, "acreate_structured_output", slow_llm)
    background = asyncio.create_task(
        extract_pending_agent_context(session_manager=sm, user_id="u", session_id="s")
    )
    stage = asyncio.create_task(
        extract_pending_agent_context(
            session_manager=sm, user_id="u", session_id="s", min_new_traces=1
        )
    )
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.gather(background, stage)

    assert calls == 1
    state = next(row for row in sm.store if row.get("kind") == TRACE_EXTRACTION_STATE_KIND)
    assert state["processed_trace_count"] == count
