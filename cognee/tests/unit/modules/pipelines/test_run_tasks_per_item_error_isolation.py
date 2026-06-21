"""Regression test: per-item error isolation in run_tasks.

run_tasks schedules every data item with ``asyncio.gather`` and then walks the
gathered values, turning any ``BaseException`` into a ``PipelineRunErrored`` and
keeping the successful results, before raising a single ``PipelineRunFailedError``
if anything failed.

That aggregation only works if ``gather`` is called with
``return_exceptions=True``. Without it, the first item to raise propagates
straight out of ``gather`` *before* the results list is even built, so the whole
``isinstance(result, BaseException)`` branch is dead code: the raw exception is
re-raised out of the generator and any successful items' results are discarded.

This test drives run_tasks with one succeeding and one failing item and asserts
the failure is aggregated (no raw raise; a PipelineRunFailedError; both items'
results preserved). On the un-fixed code the raw ValueError escapes instead.
"""

from contextlib import asynccontextmanager
import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.pipelines.exceptions import PipelineRunFailedError
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored, PipelineRunStarted
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


class _FakeEngine:
    def __init__(self, dataset):
        self._dataset = dataset

    def get_async_session(self):
        return _FakeSession(self._dataset)


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


@pytest.mark.asyncio
async def test_failing_item_is_aggregated_not_raised_raw(monkeypatch):
    dataset_id = uuid4()
    pipeline_run_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, name="dataset-1", owner_id=uuid4())
    user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())

    ok_item = SimpleNamespace(id=uuid4(), kind="ok")
    bad_item = SimpleNamespace(id=uuid4(), kind="bad")

    async def _run_item(data_item, *_args, **_kwargs):
        if data_item.kind == "bad":
            raise ValueError("boom")
        return {"run_info": SimpleNamespace(status="PipelineRunCompleted")}

    rollback_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _log_start(*_args, **_kwargs):
        return SimpleNamespace(pipeline_run_id=pipeline_run_id)

    async def _log_error(*_args, **_kwargs):
        return None

    monkeypatch.setattr(run_tasks_module, "get_relational_engine", lambda: _FakeEngine(dataset))
    monkeypatch.setattr(run_tasks_module, "generate_pipeline_id", lambda *_args: uuid4())
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_start", _log_start)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", _log_error)
    monkeypatch.setattr(run_tasks_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", _run_item)

    yielded = []
    # On the un-fixed code this raises ValueError("boom") instead of yielding.
    async for item in run_tasks_module.run_tasks(
        tasks=[Task(lambda x: x)],
        dataset_id=dataset_id,
        data=[ok_item, bad_item],
        user=user,
        pipeline_name="cognify_pipeline",
        rollback_handler=_rollback_handler,
    ):
        yielded.append(item)

    # The run is reported as errored, not crashed with the raw exception.
    assert isinstance(yielded[0], PipelineRunStarted)
    assert isinstance(yielded[-1], PipelineRunErrored)

    # Rollback receives the aggregated per-item results and a PipelineRunFailedError.
    assert len(rollback_calls) == 1
    info = rollback_calls[0]["data_ingestion_info"]
    assert isinstance(rollback_calls[0]["error"], PipelineRunFailedError)
    assert info is not None
    # Both items are represented: one success result and one per-item error.
    statuses = [entry["run_info"].status for entry in info]
    assert "PipelineRunErrored" in statuses
    assert "PipelineRunCompleted" in statuses
