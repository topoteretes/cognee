"""improve() persists agent traces in bounded windows, never as one unbounded document."""

import importlib
import types
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.cache.models import SessionAgentTraceEntry
from cognee.infrastructure.session.session_persist_watermark import TRACE_PERSIST_WATERMARK
from cognee.modules.improve import GraphCapabilities, ImproveConfig, ImproveRunInputs
from cognee.modules.improve.stages import PersistAgentTracesStage
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback
from cognee.tasks.memify.extract_agent_trace_feedbacks import extract_agent_trace_feedbacks


class FakeSessionManager:
    """In-memory trace store with the context rows the persist watermark lives in."""

    is_available = True

    def __init__(self):
        self.traces: dict[str, list[SessionAgentTraceEntry]] = {}
        self.context: dict[str, list[dict]] = {}

    def add_steps(self, session_id: str, count: int):
        steps = self.traces.setdefault(session_id, [])
        for _ in range(count):
            steps.append(
                SessionAgentTraceEntry(
                    trace_id=str(uuid4()),
                    origin_function="agent",
                    status="success",
                    session_feedback=f"{session_id} step {len(steps)}",
                )
            )

    async def get_agent_trace_count(self, *, user_id, session_id=None):
        return len(self.traces.get(session_id, []))

    async def get_agent_trace_session(self, *, user_id, session_id=None, last_n=None):
        return list(self.traces.get(session_id, []))

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


class FakeCognee:
    """Records each document handed to cognify; fails the builds listed in ``fail_builds``."""

    def __init__(self, fail_builds=()):
        self.documents: list[str] = []
        self.fail_builds = set(fail_builds)

    async def add(self, text, **kwargs):
        self.documents.append(text)

    async def cognify(self, **kwargs):
        if len(self.documents) in self.fail_builds:
            return PipelineRunErrored(
                pipeline_run_id=uuid4(),
                dataset_id=uuid4(),
                dataset_name="docs",
                error_message="rate limit exceeded",
            )
        return {}

    def steps_per_document(self) -> list[int]:
        return [len(document.split("\n\n", 1)[1].splitlines()) for document in self.documents]


@pytest.fixture
def store(monkeypatch):
    manager = FakeSessionManager()
    for module in (
        "cognee.infrastructure.session.get_session_manager",
        "cognee.tasks.memify.extract_agent_trace_feedbacks",
        "cognee.tasks.memify.cognify_agent_trace_feedback",
    ):
        monkeypatch.setattr(importlib.import_module(module), "get_session_manager", lambda: manager)
    return manager


def _use(monkeypatch, cognee_fake, **settings):
    config = ImproveConfig(**settings)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.improve.config"),
        "get_improve_config",
        lambda: config,
    )
    for name in ("add", "cognify"):
        monkeypatch.setattr(
            importlib.import_module("cognee"), name, getattr(cognee_fake, name), raising=False
        )

    async def pipeline(*, user, session_ids, raw_trace_content, last_n_steps, **kwargs):
        """The memify pipeline's two real tasks, without the dataset and pipeline plumbing."""
        extract_module = importlib.import_module(
            "cognee.tasks.memify.extract_agent_trace_feedbacks"
        )
        monkeypatch.setattr(extract_module, "session_user", types.SimpleNamespace(get=lambda: user))
        async for window in extract_agent_trace_feedbacks(
            [{}],
            session_ids=session_ids,
            raw_trace_content=raw_trace_content,
            last_n_steps=last_n_steps,
        ):
            await cognify_agent_trace_feedback(window, dataset_id=uuid4(), user=user)
        return {}

    monkeypatch.setattr(
        importlib.import_module(
            "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph"
        ),
        "persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        AsyncMock(side_effect=pipeline),
    )


def _inputs(session_ids=("s1",)):
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


async def _watermark(store, inputs, session_id="s1"):
    return await TRACE_PERSIST_WATERMARK.read_count(store, str(inputs.user.id), session_id)


@pytest.mark.asyncio
async def test_backlog_is_persisted_in_documents_of_at_most_the_window_size(store, monkeypatch):
    store.add_steps("s1", 95)
    cognee_fake = FakeCognee()
    _use(monkeypatch, cognee_fake, trace_window_steps=30)
    inputs = _inputs()

    result = await PersistAgentTracesStage().run(inputs)

    assert result.status == "completed"
    assert cognee_fake.steps_per_document() == [30, 30, 30, 5]
    assert await _watermark(store, inputs) == 95


@pytest.mark.asyncio
async def test_failed_window_keeps_earlier_windows_and_stops_the_stage(store, monkeypatch):
    store.add_steps("s1", 95)
    cognee_fake = FakeCognee(fail_builds={2})
    _use(monkeypatch, cognee_fake, trace_window_steps=30)
    inputs = _inputs()

    result = await PersistAgentTracesStage().run(inputs)

    assert result.status == "errored"
    assert cognee_fake.steps_per_document() == [30, 30]
    assert await _watermark(store, inputs) == 30


@pytest.mark.asyncio
async def test_window_cap_defers_the_rest_of_the_backlog_to_the_next_run(store, monkeypatch):
    store.add_steps("a", 95)
    store.add_steps("b", 95)
    cognee_fake = FakeCognee()
    _use(monkeypatch, cognee_fake, trace_window_steps=30, trace_windows_per_run=3)
    inputs = _inputs(session_ids=("a", "b"))

    first = await PersistAgentTracesStage().run(inputs)

    assert first.status == "completed"
    assert first.counts["windows"] == 3
    assert len(cognee_fake.documents) == 3
    assert await _watermark(store, inputs, "a") + await _watermark(store, inputs, "b") == 90

    for _ in range(3):
        await PersistAgentTracesStage().run(inputs)

    assert await _watermark(store, inputs, "a") == 95
    assert await _watermark(store, inputs, "b") == 95
    assert max(cognee_fake.steps_per_document()) == 30


@pytest.mark.parametrize("field", ["trace_window_steps", "trace_windows_per_run"])
def test_window_settings_must_be_positive(field):
    with pytest.raises(ValueError):
        ImproveConfig(**{field: 0})
