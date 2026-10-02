"""Pipeline and task telemetry events (SDK-775).

Every event of a run carries ``pipeline_run_id``; error events carry the
exception class and never its message; cancellation and a closed generator
still emit a terminal event instead of leaving a silent ``Started``.
"""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.operations import run_tasks_base as base_module
from cognee.modules.pipelines.operations import run_tasks_with_telemetry as telemetry_module
from cognee.modules.pipelines.tasks.task import Task

SETTINGS = {
    "llm": {"provider": "openai", "model": "openai/gpt-5-mini"},
    "embedding": {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
    "graph_extractor": "llm",
    "graph": {"provider": "kuzu", "url": "/tmp/graph"},
    "vector": {"provider": "lancedb", "url": "/tmp/vector"},
    "relational": {"provider": "sqlite", "url": "/tmp/db"},
}
# Would leak a dataset name if any event carried the message.
SECRET_MESSAGE = "Dataset 'customer-secrets-2026' not found."
USER = SimpleNamespace(id=uuid4(), tenant_id=None)


@pytest.fixture
def events(monkeypatch):
    captured = []

    def capture(event_name, user=None, additional_properties=None, **_kwargs):
        captured.append((event_name, dict(additional_properties or {})))

    monkeypatch.setattr(telemetry_module, "send_telemetry", capture)
    monkeypatch.setattr(base_module, "send_telemetry", capture)
    monkeypatch.setattr(telemetry_module, "get_current_settings", lambda: dict(SETTINGS))
    return captured


def _pipeline_events(events):
    return [(name, props) for name, props in events if name.startswith("Pipeline Run")]


def _task_events(events, suffix):
    return [(name, props) for name, props in events if name.endswith(f"Task {suffix}")]


async def _drain(tasks, ctx):
    async for _ in telemetry_module.run_tasks_with_telemetry(
        tasks, [1], USER, "cognify_pipeline", ctx=ctx
    ):
        pass


@pytest.mark.asyncio
async def test_started_and_completed_carry_run_id_and_provider_stack(events):
    run_id = uuid4()

    async def double(data):
        return [item * 2 for item in data]

    await _drain([Task(double)], PipelineContext(pipeline_run_id=run_id))

    names = [name for name, _ in _pipeline_events(events)]
    assert names == ["Pipeline Run Started", "Pipeline Run Completed"]
    for _, props in _pipeline_events(events):
        assert props["pipeline_run_id"] == str(run_id)
        assert props["pipeline_name"] == "cognify_pipeline"
        assert props["embedding"] == SETTINGS["embedding"]
        assert props["graph_extractor"] == "llm"
        assert "exception_type" not in props
    # Task events join to the same run.
    for _, props in _task_events(events, "Started") + _task_events(events, "Completed"):
        assert props["pipeline_run_id"] == str(run_id)
        assert props["task_name"] == "double"


@pytest.mark.asyncio
async def test_errored_carries_exception_type_and_never_the_message(events):
    async def explode(data):
        raise ValueError(SECRET_MESSAGE)

    with pytest.raises(ValueError):
        await _drain([Task(explode)], PipelineContext(pipeline_run_id=uuid4()))

    (pipeline_errored,) = [p for n, p in _pipeline_events(events) if n == "Pipeline Run Errored"]
    assert pipeline_errored["exception_type"] == "ValueError"
    ((_, task_errored),) = _task_events(events, "Errored")
    assert task_errored["exception_type"] == "ValueError"
    assert task_errored["task_name"] == "explode"
    for _, props in events:
        assert SECRET_MESSAGE not in repr(props)


@pytest.mark.asyncio
async def test_root_cause_is_reported_through_a_wrapping_error(events):
    class Wrapper(Exception):
        pass

    async def explode(data):
        wrapper = Wrapper("Pipeline run failed.")
        wrapper.first_error = KeyError("root")
        raise wrapper

    with pytest.raises(Wrapper):
        await _drain([Task(explode)], PipelineContext(pipeline_run_id=uuid4()))

    (errored,) = [p for n, p in _pipeline_events(events) if n == "Pipeline Run Errored"]
    assert errored["exception_type"] == "KeyError"


@pytest.mark.asyncio
async def test_cancelled_run_emits_a_terminal_event(events):
    async def cancelled(data):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _drain([Task(cancelled)], PipelineContext(pipeline_run_id=uuid4()))

    names = [name for name, _ in _pipeline_events(events)]
    assert names == ["Pipeline Run Started", "Pipeline Run Errored"]
    (errored,) = [p for n, p in _pipeline_events(events) if n == "Pipeline Run Errored"]
    assert errored["exception_type"] == "CancelledError"
    ((_, task_errored),) = _task_events(events, "Errored")
    assert task_errored["exception_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_closed_generator_emits_a_terminal_event(events):
    """A consumer that stops iterating closes the generator with GeneratorExit —
    a BaseException a bare ``except Exception`` never saw."""

    async def stream(data):
        yield 1
        yield 2

    generator = telemetry_module.run_tasks_with_telemetry(
        [Task(stream)], [1], USER, "cognify_pipeline", ctx=PipelineContext(pipeline_run_id=uuid4())
    )
    await generator.__anext__()
    await generator.aclose()

    names = [name for name, _ in _pipeline_events(events)]
    assert names == ["Pipeline Run Started", "Pipeline Run Errored"]
    (errored,) = [p for n, p in _pipeline_events(events) if n == "Pipeline Run Errored"]
    assert errored["exception_type"] == "GeneratorExit"


@pytest.mark.asyncio
async def test_no_context_means_no_run_id_but_the_events_still_flow(events):
    async def identity(data):
        return data

    await _drain([Task(identity)], ctx=None)

    for _, props in _pipeline_events(events):
        assert "pipeline_run_id" not in props
        assert props["pipeline_name"] == "cognify_pipeline"
