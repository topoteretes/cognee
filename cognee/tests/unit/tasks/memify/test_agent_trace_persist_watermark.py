"""Watermark-based incremental agent-trace persistence tests."""

import sys
import uuid
from types import SimpleNamespace

import pytest

import cognee
from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.session.agent_trace_persist_watermark import (
    AgentTracePersistWindow,
    get_persisted_trace_count,
    save_persisted_trace_count,
)
from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback
from cognee.tasks.memify.extract_agent_trace_feedbacks import extract_agent_trace_feedbacks

SESSION = "trace_watermark_test_session"


class FakeSessionManager:
    is_available = True

    def __init__(self):
        self.traces: dict[tuple[str, str], list[str]] = {}
        self.context: dict[tuple[str, str], list[dict]] = {}

    def add_trace(self, user_id: str, session_id: str, feedback: str):
        self.traces.setdefault((user_id, session_id), []).append(feedback)

    async def get_agent_trace_count(self, *, user_id, session_id=None):
        return len(self.traces.get((user_id, session_id), []))

    async def get_agent_trace_feedback(self, *, user_id, session_id=None, last_n=None):
        traces = self.traces.get((user_id, session_id), [])
        return list(traces if last_n is None else traces[-last_n:])

    async def get_session_context_entries(self, *, user_id, session_id=None):
        return list(self.context.get((user_id, session_id), []))

    async def update_session_context_entry(self, *, user_id, entry_id, merge, session_id=None):
        for row in self.context.get((user_id, session_id), []):
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id=None):
        self.context.setdefault((user_id, session_id), []).append(dict(entry_dump))
        return True


@pytest.fixture
def user():
    fake_user = SimpleNamespace(id=uuid.uuid4())
    token = session_user.set(fake_user)
    yield fake_user
    session_user.reset(token)


@pytest.fixture
def manager(monkeypatch):
    extract_module = sys.modules["cognee.tasks.memify.extract_agent_trace_feedbacks"]
    cognify_module = sys.modules["cognee.tasks.memify.cognify_agent_trace_feedback"]
    fake = FakeSessionManager()
    monkeypatch.setattr(extract_module, "get_session_manager", lambda: fake)
    monkeypatch.setattr(cognify_module, "get_session_manager", lambda: fake)
    return fake


async def _extract_windows() -> list[AgentTracePersistWindow]:
    return [
        window
        async for window in extract_agent_trace_feedbacks(
            [{}], session_ids=[SESSION], incremental=True
        )
    ]


@pytest.mark.asyncio
async def test_incremental_trace_persistence_processes_each_step_once(user, manager, monkeypatch):
    user_id = str(user.id)
    persisted_texts: list[str] = []

    async def fake_add(text, *args, **kwargs):
        persisted_texts.append(text)

    async def fake_cognify(*args, **kwargs):
        return None

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", fake_cognify)

    manager.add_trace(user_id, SESSION, "step-1")
    manager.add_trace(user_id, SESSION, "step-2")
    first_windows = await _extract_windows()
    assert len(first_windows) == 1
    await cognify_agent_trace_feedback(first_windows[0], dataset_id=uuid.uuid4(), user=user)

    assert "step-1" in persisted_texts[0]
    assert "step-2" in persisted_texts[0]
    assert await get_persisted_trace_count(manager, user_id, SESSION) == 2
    assert await _extract_windows() == []

    manager.add_trace(user_id, SESSION, "step-3")
    next_windows = await _extract_windows()
    assert len(next_windows) == 1
    assert "step-1" not in next_windows[0].text
    assert "step-2" not in next_windows[0].text
    assert "step-3" in next_windows[0].text
    await cognify_agent_trace_feedback(next_windows[0], dataset_id=uuid.uuid4(), user=user)

    combined = "".join(persisted_texts)
    assert combined.count("step-1") == 1
    assert combined.count("step-2") == 1
    assert combined.count("step-3") == 1
    assert await get_persisted_trace_count(manager, user_id, SESSION) == 3


@pytest.mark.asyncio
async def test_failed_trace_persistence_keeps_window_pending(user, manager, monkeypatch):
    user_id = str(user.id)
    manager.add_trace(user_id, SESSION, "retry-me")
    window = (await _extract_windows())[0]

    async def fake_add(*args, **kwargs):
        return None

    async def failing_cognify(*args, **kwargs):
        raise RuntimeError("temporary LLM failure")

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", failing_cognify)

    with pytest.raises(CogneeSystemError):
        await cognify_agent_trace_feedback(window, dataset_id=uuid.uuid4(), user=user)

    assert await get_persisted_trace_count(manager, user_id, SESSION) == 0
    retry_windows = await _extract_windows()
    assert len(retry_windows) == 1
    assert "retry-me" in retry_windows[0].text


@pytest.mark.asyncio
async def test_errored_cognify_result_keeps_trace_window_pending(user, manager, monkeypatch):
    from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored

    user_id = str(user.id)
    manager.add_trace(user_id, SESSION, "retry-errored-run")
    window = (await _extract_windows())[0]

    async def fake_add(*args, **kwargs):
        return None

    async def errored_cognify(*args, **kwargs):
        return {
            "dataset": PipelineRunErrored(
                pipeline_run_id=uuid.uuid4(),
                dataset_id=uuid.uuid4(),
                dataset_name="dataset",
                error_class="RuntimeError",
                error_message="temporary failure",
            )
        }

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", errored_cognify)

    await cognify_agent_trace_feedback(window, dataset_id=uuid.uuid4(), user=user)

    assert await get_persisted_trace_count(manager, user_id, SESSION) == 0
    assert len(await _extract_windows()) == 1


@pytest.mark.asyncio
async def test_empty_feedback_advances_watermark_without_cognify(user, manager, monkeypatch):
    user_id = str(user.id)
    manager.add_trace(user_id, SESSION, "   ")
    window = (await _extract_windows())[0]

    async def unexpected_call(*args, **kwargs):
        raise AssertionError("empty feedback must not trigger graph ingestion")

    monkeypatch.setattr(cognee, "add", unexpected_call)
    monkeypatch.setattr(cognee, "cognify", unexpected_call)

    await cognify_agent_trace_feedback(window, dataset_id=uuid.uuid4(), user=user)

    assert await get_persisted_trace_count(manager, user_id, SESSION) == 1
    assert await _extract_windows() == []


@pytest.mark.asyncio
async def test_stale_trace_watermark_restarts_from_current_history(user, manager):
    user_id = str(user.id)
    manager.add_trace(user_id, SESSION, "rebuilt-step")
    await save_persisted_trace_count(manager, user_id, SESSION, 10)

    windows = await _extract_windows()

    assert len(windows) == 1
    assert "rebuilt-step" in windows[0].text
    assert windows[0].persisted_trace_count == 1
