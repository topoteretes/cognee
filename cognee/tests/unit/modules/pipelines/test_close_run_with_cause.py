"""A run closed because of a known failure records that failure, not the closure."""

import importlib
from contextlib import aclosing, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunStarted
from cognee.modules.pipelines.tasks.task import Task

run_tasks_module = importlib.import_module("cognee.modules.pipelines.operations.run_tasks")


class _FakeSession:
    def __init__(self, dataset):
        self._dataset = dataset

    async def get(self, _model, _dataset_id):
        return self._dataset

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


@pytest.fixture
def paused_run(monkeypatch):
    """A run_tasks run held at its start event, and the error rows it writes."""
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    run_id = uuid4()
    log_error = AsyncMock()

    monkeypatch.setattr(
        run_tasks_module,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=lambda: _FakeSession(dataset)),
    )
    monkeypatch.setattr(run_tasks_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(
        run_tasks_module,
        "log_pipeline_run_start",
        AsyncMock(return_value=SimpleNamespace(pipeline_run_id=run_id)),
    )
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", log_error)
    monkeypatch.setattr(run_tasks_module, "send_telemetry", lambda *args, **kwargs: None)

    def start():
        return run_tasks_module.run_tasks([Task(lambda x: x)], dataset.id, [1], user)

    return start, log_error


@pytest.mark.asyncio
async def test_the_cause_is_recorded_on_the_closed_run(paused_run):
    start, log_error = paused_run
    run = start()
    async with aclosing(run):
        assert isinstance(await anext(run), PipelineRunStarted)
        cause = RuntimeError("vector store unreachable")
        await run_tasks_module.close_run_with_cause(run, cause)

    log_error.assert_awaited_once()
    assert log_error.await_args.args[4] is cause


@pytest.mark.asyncio
async def test_a_plain_close_still_records_the_closure(paused_run):
    start, log_error = paused_run
    run = start()
    assert isinstance(await anext(run), PipelineRunStarted)
    await run.aclose()

    log_error.assert_awaited_once()
    assert isinstance(log_error.await_args.args[4], GeneratorExit)
    assert run_tasks_module._close_cause.get() is None
