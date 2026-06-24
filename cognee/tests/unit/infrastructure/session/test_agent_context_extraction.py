"""Unit tests for the live (deterministic, no-LLM) agent-context extraction pass.

Uses a fake session manager so it runs anywhere; verifies that only errored steps produce a
failure_lessons lesson, that the lesson is linked to its trace, and the fail-open contract.
"""

import pytest

from cognee.infrastructure.session.agent_context_extraction import (
    LIVE_FAILURE_CONFIDENCE,
    build_live_agent_candidates,
    extract_live_agent_context,
)


class FakeSessionManager:
    """In-memory stand-in for the context CRUD surface used by the applier."""

    def __init__(self):
        self.store = []

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
