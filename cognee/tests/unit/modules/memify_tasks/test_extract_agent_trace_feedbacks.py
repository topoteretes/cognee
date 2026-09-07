"""Watermark-bounded extraction of agent trace steps (improve stage 3).

The extractor yields one ``TracePersistWindow`` per session holding only the
trace steps above the session's trace persist watermark; ``last_n_steps=None``
means "everything above the watermark", never "everything stored".
"""

import sys
from unittest.mock import MagicMock

import pytest

from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.databases.cache.models import SessionAgentTraceEntry
from cognee.infrastructure.session.session_persist_watermark import (
    TRACE_PERSIST_STATE_ID,
    TRACE_PERSIST_STATE_KIND,
    TracePersistWindow,
    get_persisted_trace_count,
    save_persisted_trace_count,
)
from cognee.modules.users.models import User
from cognee.tasks.memify.extract_agent_trace_feedbacks import (
    extract_agent_trace_feedbacks,
    resolve_trace_window_size,
)

extract_agent_trace_feedbacks_module = sys.modules[
    "cognee.tasks.memify.extract_agent_trace_feedbacks"
]

USER_ID = "test-user-123"


def _trace_entry(method_return_value, session_feedback=""):
    """Build a SessionAgentTraceEntry as get_agent_trace_session returns in production."""
    return SessionAgentTraceEntry(
        trace_id="t1",
        origin_function="traced_agent",
        status="success",
        method_return_value=method_return_value,
        session_feedback=session_feedback,
    )


class FakeSessionManager:
    """In-memory stand-in exposing the trace-read + watermark surface the task uses.

    ``last_n`` follows the adapters: the most recent N steps, oldest first.
    """

    def __init__(self, is_available: bool = True):
        self.is_available = is_available
        self.traces: dict[str, list[SessionAgentTraceEntry]] = {}
        self.context: dict[str, list[dict]] = {}
        self.feedback_last_n_calls: list = []
        self.session_last_n_calls: list = []
        self.failing_sessions: set[str] = set()

    def add_step(self, session_id: str, feedback: str = "", return_value=None):
        self.traces.setdefault(session_id, []).append(
            _trace_entry(return_value, session_feedback=feedback)
        )

    async def get_agent_trace_count(self, *, user_id, session_id=None):
        if session_id in self.failing_sessions:
            raise RuntimeError("SessionManager error")
        return len(self.traces.get(session_id, []))

    async def get_agent_trace_session(self, *, user_id, session_id=None, last_n=None):
        self.session_last_n_calls.append(last_n)
        entries = list(self.traces.get(session_id, []))
        return entries[-last_n:] if last_n is not None else entries

    async def get_agent_trace_feedback(self, *, user_id, session_id=None, last_n=None):
        self.feedback_last_n_calls.append(last_n)
        entries = await self.get_agent_trace_session(
            user_id=user_id, session_id=session_id, last_n=last_n
        )
        self.session_last_n_calls.pop()
        return [entry.session_feedback for entry in entries]

    async def get_session_context_entries(self, *, user_id, session_id=None):
        return list(self.context.get(session_id, []))

    async def update_session_context_entry(self, *, user_id, entry_id, merge, session_id=None):
        for row in self.context.get(session_id, []):
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id=None):
        self.context.setdefault(session_id, []).append(dict(entry_dump))
        return True


@pytest.fixture
def mock_user():
    user = MagicMock(spec=User)
    user.id = USER_ID
    return user


@pytest.fixture
def manager(mock_user, monkeypatch):
    fake = FakeSessionManager()
    session_user = MagicMock()
    session_user.get.return_value = mock_user
    monkeypatch.setattr(extract_agent_trace_feedbacks_module, "session_user", session_user)
    monkeypatch.setattr(extract_agent_trace_feedbacks_module, "get_session_manager", lambda: fake)
    return fake


async def _extract(session_ids, **kwargs) -> list[TracePersistWindow]:
    return [
        window
        async for window in extract_agent_trace_feedbacks([{}], session_ids=session_ids, **kwargs)
    ]


# ------------------------------------------------------------------ window arithmetic


@pytest.mark.parametrize(
    ("total", "persisted", "last_n_steps", "expected"),
    [
        (3, 0, None, 3),  # fresh session: everything
        (5, 3, None, 2),  # above the watermark only
        (5, 5, None, 0),  # fully persisted
        (0, 0, None, 0),  # nothing stored
        (2, 10, None, 2),  # stale watermark (session rebuilt): start over
        (5, 1, 2, 2),  # explicit cap bounds the pending window
        (5, 4, 2, 1),  # cap larger than pending: pending wins
        (5, 0, 0, 0),  # zero cap: nothing
    ],
)
def test_resolve_trace_window_size(total, persisted, last_n_steps, expected):
    assert resolve_trace_window_size(total, persisted, last_n_steps) == expected


# ------------------------------------------------------------------ extraction contract


@pytest.mark.asyncio
async def test_fresh_session_yields_all_feedback_above_watermark(manager):
    manager.add_step("trace_session", feedback="draft plan succeeded.")
    manager.add_step("trace_session", feedback="   ")
    manager.add_step("trace_session", feedback="write_summary failed.")

    windows = await _extract(["trace_session"])

    assert len(windows) == 1
    window = windows[0]
    assert (
        window.text == "Session ID: trace_session\n\ndraft plan succeeded.\nwrite_summary failed."
    )
    assert window.user_id == USER_ID
    assert window.session_id == "trace_session"
    assert window.persisted_trace_count == 3
    # Only the pending steps are read: every step above the (empty) watermark.
    assert manager.feedback_last_n_calls == [3]
    # The extractor never advances the watermark itself for a non-empty window.
    assert await get_persisted_trace_count(manager, USER_ID, "trace_session") == 0


@pytest.mark.asyncio
async def test_watermark_skips_already_persisted_steps(manager):
    for index in range(5):
        manager.add_step("s", feedback=f"step {index}")
    await save_persisted_trace_count(manager, USER_ID, "s", 3)

    windows = await _extract(["s"])

    assert len(windows) == 1
    assert manager.feedback_last_n_calls == [2]
    assert "step 2" not in windows[0].text
    assert "step 3" in windows[0].text and "step 4" in windows[0].text
    assert windows[0].persisted_trace_count == 5


@pytest.mark.asyncio
async def test_fully_persisted_session_yields_nothing_and_reads_no_steps(manager):
    manager.add_step("s", feedback="done")
    await save_persisted_trace_count(manager, USER_ID, "s", 1)

    assert await _extract(["s"]) == []
    assert manager.feedback_last_n_calls == []
    assert manager.session_last_n_calls == []


@pytest.mark.asyncio
async def test_session_without_steps_yields_nothing(manager):
    assert await _extract(["empty"]) == []
    assert manager.feedback_last_n_calls == []


@pytest.mark.asyncio
async def test_stale_watermark_restarts_from_the_beginning(manager):
    manager.add_step("s", feedback="rebuilt step")
    await save_persisted_trace_count(manager, USER_ID, "s", 10)

    windows = await _extract(["s"])

    assert len(windows) == 1
    assert "rebuilt step" in windows[0].text
    assert manager.feedback_last_n_calls == [1]
    assert windows[0].persisted_trace_count == 1


@pytest.mark.asyncio
async def test_last_n_steps_caps_the_pending_window(manager):
    for index in range(5):
        manager.add_step("s", feedback=f"step {index}")
    await save_persisted_trace_count(manager, USER_ID, "s", 1)

    windows = await _extract(["s"], last_n_steps=2)

    assert manager.feedback_last_n_calls == [2]
    assert windows[0].text == "Session ID: s\n\nstep 3\nstep 4"
    # The watermark still advances to the total once cognified: steps under the
    # explicit cap are deliberately left behind, as last_n always did.
    assert windows[0].persisted_trace_count == 5


@pytest.mark.asyncio
async def test_multiple_sessions_yield_one_window_each(manager):
    manager.add_step("session1", feedback="step completed.")
    manager.add_step("session2", feedback="step completed.")

    windows = await _extract(["session1", "session2"])

    assert [window.session_id for window in windows] == ["session1", "session2"]
    assert manager.session_last_n_calls == []


@pytest.mark.asyncio
async def test_window_without_persistable_content_advances_watermark_without_yield(manager):
    manager.add_step("empty_session", feedback="   ")
    manager.add_step("empty_session", feedback="")

    assert await _extract(["empty_session"]) == []
    # Nothing to cognify, so there is no success to wait on: mark the window done
    # instead of re-reading it on every run.
    assert await get_persisted_trace_count(manager, USER_ID, "empty_session") == 2
    state = manager.context["empty_session"][0]
    assert state["id"] == TRACE_PERSIST_STATE_ID
    assert state["kind"] == TRACE_PERSIST_STATE_KIND


@pytest.mark.asyncio
async def test_no_session_ids_yields_nothing(manager):
    manager.add_step("s", feedback="step completed.")

    assert await _extract(None) == []
    assert manager.feedback_last_n_calls == []


@pytest.mark.asyncio
async def test_session_manager_unavailable_raises(manager):
    manager.is_available = False

    with pytest.raises(CogneeSystemError) as exc_info:
        await _extract(["trace_session"])

    assert "SessionManager not available" in str(exc_info.value)


@pytest.mark.asyncio
async def test_continues_when_one_session_fails(manager):
    manager.add_step("session1", feedback="first feedback")
    manager.add_step("session2", feedback="ignored")
    manager.add_step("session3", feedback="third feedback")
    manager.failing_sessions.add("session2")

    windows = await _extract(["session1", "session2", "session3"])

    assert [window.text for window in windows] == [
        "Session ID: session1\n\nfirst feedback",
        "Session ID: session3\n\nthird feedback",
    ]


@pytest.mark.asyncio
async def test_raw_return_values_are_extracted_above_the_watermark(manager):
    manager.add_step("trace_session", return_value="already persisted")
    manager.add_step("trace_session", return_value="draft ready")
    manager.add_step("trace_session", return_value={"summary": "done", "steps": 2})
    manager.add_step("trace_session", return_value="   ")
    manager.add_step("trace_session", return_value=None)
    await save_persisted_trace_count(manager, USER_ID, "trace_session", 1)

    windows = await _extract(["trace_session"], raw_trace_content=True)

    assert [window.text for window in windows] == [
        'Session ID: trace_session\n\ndraft ready\n{"steps": 2, "summary": "done"}'
    ]
    assert windows[0].persisted_trace_count == 5
    assert manager.session_last_n_calls == [4]
    assert manager.feedback_last_n_calls == []


@pytest.mark.asyncio
async def test_raw_return_values_respect_last_n_steps_cap(manager):
    for value in ("first return", "second return", "third return"):
        manager.add_step("trace_session", return_value=value)

    windows = await _extract(["trace_session"], raw_trace_content=True, last_n_steps=2)

    assert [window.text for window in windows] == [
        "Session ID: trace_session\n\nsecond return\nthird return"
    ]
    assert manager.session_last_n_calls == [2]


@pytest.mark.asyncio
async def test_rejects_non_boolean_raw_trace_content(manager):
    manager.add_step("trace_session", feedback="draft plan succeeded.")

    with pytest.raises(CogneeSystemError, match="raw_trace_content must be a boolean"):
        await _extract(["trace_session"], raw_trace_content="yes")
