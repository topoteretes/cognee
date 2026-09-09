"""Watermark-based incremental trace persistence (SDK-593).

Mirrors the QA persist watermark contract:
1. Completeness — every trace step is ingested once.
2. Incrementality — already-persisted steps are never re-ingested.
3. Retry safety — a failed cognify leaves the watermark untouched.
"""

import sys
import uuid
from types import SimpleNamespace

import pytest

import cognee
from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.databases.cache.models import SessionAgentTraceEntry
from cognee.infrastructure.session.session_trace_persist_watermark import (
    TracePersistWindow,
    get_persisted_trace_count,
    save_persisted_trace_count,
)
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback
from cognee.tasks.memify.extract_agent_trace_feedbacks import extract_agent_trace_feedbacks

SESSION = "trace_watermark_test_session"


class FakeSessionManager:
    is_available = True

    def __init__(self):
        self.traces: dict[tuple[str, str], list[SessionAgentTraceEntry]] = {}
        self.context: dict[tuple[str, str], list[dict]] = {}

    def add_step(self, user_id, session_id, feedback):
        self.traces.setdefault((user_id, session_id), []).append(
            SessionAgentTraceEntry(
                trace_id=str(uuid.uuid4()),
                origin_function="Bash",
                status="success",
                session_feedback=feedback,
            )
        )

    async def get_agent_trace_session(self, *, user_id, session_id=None, last_n=None):
        entries = list(self.traces.get((user_id, session_id), []))
        return entries[-last_n:] if last_n else entries

    async def get_agent_trace_feedback(self, *, user_id, session_id=None, last_n=None):
        entries = await self.get_agent_trace_session(
            user_id=user_id, session_id=session_id, last_n=last_n
        )
        return [entry.session_feedback for entry in entries]

    async def get_agent_trace_count(self, *, user_id, session_id=None):
        return len(self.traces.get((user_id, session_id), []))

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
def manager(user, monkeypatch):
    extract_module = sys.modules["cognee.tasks.memify.extract_agent_trace_feedbacks"]
    cognify_module = sys.modules["cognee.tasks.memify.cognify_agent_trace_feedback"]
    fake = FakeSessionManager()
    monkeypatch.setattr(extract_module, "get_session_manager", lambda: fake)
    monkeypatch.setattr(cognify_module, "get_session_manager", lambda: fake)
    return fake


async def _extract(session_ids) -> list[TracePersistWindow]:
    return [w async for w in extract_agent_trace_feedbacks([{}], session_ids=session_ids)]


@pytest.mark.asyncio
async def test_fresh_session_extracts_all_steps(user, manager):
    user_id = str(user.id)
    manager.add_step(user_id, SESSION, "ran tests")
    manager.add_step(user_id, SESSION, "edited file")

    windows = await _extract([SESSION])

    assert len(windows) == 1
    assert "ran tests" in windows[0].text and "edited file" in windows[0].text
    assert windows[0].persisted_trace_count == 2


@pytest.mark.asyncio
async def test_watermark_skips_already_persisted_steps(user, manager):
    user_id = str(user.id)
    for index in range(5):
        manager.add_step(user_id, SESSION, f"step{index}")
    await save_persisted_trace_count(manager, user_id, SESSION, 3)

    windows = await _extract([SESSION])

    assert len(windows) == 1
    for old_index in range(3):
        assert f"step{old_index}" not in windows[0].text
    assert "step3" in windows[0].text and "step4" in windows[0].text
    assert windows[0].persisted_trace_count == 5


@pytest.mark.asyncio
async def test_fully_persisted_session_yields_nothing(user, manager):
    user_id = str(user.id)
    manager.add_step(user_id, SESSION, "step")
    await save_persisted_trace_count(manager, user_id, SESSION, 1)

    assert await _extract([SESSION]) == []


@pytest.mark.asyncio
async def test_watermark_roundtrip_updates_one_row(user, manager):
    user_id = str(user.id)
    assert await get_persisted_trace_count(manager, user_id, SESSION) == 0
    await save_persisted_trace_count(manager, user_id, SESSION, 4)
    await save_persisted_trace_count(manager, user_id, SESSION, 9)
    assert await get_persisted_trace_count(manager, user_id, SESSION) == 9
    assert len(manager.context[(user_id, SESSION)]) == 1


@pytest.mark.asyncio
async def test_cognify_advances_watermark_on_success(user, manager, monkeypatch):
    user_id = str(user.id)
    calls = []

    async def fake_add(text, **kwargs):
        calls.append(("add", text))

    async def fake_cognify(**kwargs):
        calls.append(("cognify",))
        return {}

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", fake_cognify)

    window = TracePersistWindow(
        user_id=user_id, session_id=SESSION, text="Session ID: s\n\nstep", persisted_trace_count=3
    )
    await cognify_agent_trace_feedback([window], dataset_id=uuid.uuid4(), user=user)

    assert [c[0] for c in calls] == ["add", "cognify"]
    assert await get_persisted_trace_count(manager, user_id, SESSION) == 3


@pytest.mark.asyncio
async def test_cognify_keeps_watermark_on_exception(user, manager, monkeypatch):
    user_id = str(user.id)

    async def fake_add(*args, **kwargs):
        return None

    async def failing_cognify(**kwargs):
        raise RuntimeError("LLM exploded")

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", failing_cognify)

    window = TracePersistWindow(
        user_id=user_id, session_id=SESSION, text="Session ID: s\n\nstep", persisted_trace_count=3
    )
    with pytest.raises(CogneeSystemError):
        await cognify_agent_trace_feedback(window, dataset_id=uuid.uuid4(), user=user)

    assert await get_persisted_trace_count(manager, user_id, SESSION) == 0


@pytest.mark.asyncio
async def test_cognify_keeps_watermark_on_errored_run_info(user, manager, monkeypatch):
    user_id = str(user.id)

    async def fake_add(*args, **kwargs):
        return None

    async def errored_cognify(**kwargs):
        return {
            "ds": PipelineRunErrored(
                pipeline_run_id=uuid.uuid4(),
                dataset_id=uuid.uuid4(),
                dataset_name="ds",
                error_class="AuthenticationError",
                error_message="bad key",
            )
        }

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", errored_cognify)

    window = TracePersistWindow(
        user_id=user_id, session_id=SESSION, text="Session ID: s\n\nstep", persisted_trace_count=3
    )
    await cognify_agent_trace_feedback(window, dataset_id=uuid.uuid4(), user=user)

    assert await get_persisted_trace_count(manager, user_id, SESSION) == 0


@pytest.mark.asyncio
async def test_empty_window_only_advances_the_watermark(user, manager, monkeypatch):
    """Steps with no content are not worth a cognify, but must not be re-read forever."""
    user_id = str(user.id)

    async def must_not_add(*args, **kwargs):
        raise AssertionError("add() must not run for an empty window")

    monkeypatch.setattr(cognee, "add", must_not_add)

    window = TracePersistWindow(
        user_id=user_id, session_id=SESSION, text="", persisted_trace_count=2
    )
    await cognify_agent_trace_feedback(window, dataset_id=uuid.uuid4(), user=user)

    assert await get_persisted_trace_count(manager, user_id, SESSION) == 2


@pytest.mark.asyncio
async def test_extract_then_cognify_then_extract_is_incremental(user, manager, monkeypatch):
    """The end-to-end contract: a second improve() over new steps ingests only those."""
    user_id = str(user.id)
    added = []

    async def fake_add(text, **kwargs):
        added.append(text)

    async def fake_cognify(**kwargs):
        return {}

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", fake_cognify)
    dataset_id = uuid.uuid4()

    manager.add_step(user_id, SESSION, "first")
    for window in await _extract([SESSION]):
        await cognify_agent_trace_feedback(window, dataset_id=dataset_id, user=user)

    assert await _extract([SESSION]) == []  # nothing new, nothing yielded

    manager.add_step(user_id, SESSION, "second")
    for window in await _extract([SESSION]):
        await cognify_agent_trace_feedback(window, dataset_id=dataset_id, user=user)

    assert len(added) == 2
    assert "first" in added[0] and "second" not in added[0]
    assert "second" in added[1] and "first" not in added[1]
    assert await get_persisted_trace_count(manager, user_id, SESSION) == 2
