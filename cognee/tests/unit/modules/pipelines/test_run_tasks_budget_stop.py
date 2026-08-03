"""Budget-exhaustion during cognify stops cleanly and keeps already-cognified data.

When an LLM budget/quota is exhausted mid-run, run_tasks must NOT roll back (so the
graph stays queryable), must NOT mark the run errored, and must report how much is
left — a later cognify resumes the remaining documents. See CLO-421.
"""

import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
)
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
async def test_budget_exhaustion_stops_cleanly_without_rollback(monkeypatch):
    dataset_id = uuid4()
    pipeline_run_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, name="dataset-1", owner_id=uuid4())
    user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
    data_item = SimpleNamespace(id=uuid4())

    # The item hits the budget wall (post-#4321 shape: a 402 LLMPaymentRequiredError).
    async def _budget_item(*_args, **_kwargs):
        raise LLMPaymentRequiredError()

    # Three documents in the dataset; one already cognified before the stop.
    def _data(completed: bool):
        status = (
            {"cognify_pipeline": {str(dataset_id): "DATA_ITEM_PROCESSING_COMPLETED"}}
            if completed
            else {}
        )
        return SimpleNamespace(pipeline_status=status)

    async def _get_dataset_data(_dataset_id):
        return [_data(True), _data(False), _data(False)]

    rollback_calls = []
    complete_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _log_start(*_args, **_kwargs):
        return SimpleNamespace(pipeline_run_id=pipeline_run_id)

    async def _log_complete(*_args, **kwargs):
        complete_calls.append(kwargs)

    async def _log_error(*_args, **_kwargs):
        return None

    monkeypatch.setattr(run_tasks_module, "get_relational_engine", lambda: _FakeEngine(dataset))
    monkeypatch.setattr(run_tasks_module, "generate_pipeline_id", lambda *_args: uuid4())
    monkeypatch.setattr(run_tasks_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", _budget_item)
    monkeypatch.setattr(run_tasks_module, "get_dataset_data", _get_dataset_data)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_start", _log_start)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_complete", _log_complete)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", _log_error)

    yielded = []
    async for item in run_tasks_module.run_tasks(
        tasks=[Task(lambda x: x)],
        dataset_id=dataset_id,
        data=[data_item],
        user=user,
        pipeline_name="cognify_pipeline",
        rollback_handler=_rollback_handler,
    ):
        yielded.append(item)

    # Terminal event is COMPLETED (a clean stop), NOT errored.
    assert isinstance(yielded[0], PipelineRunStarted)
    assert isinstance(yielded[-1], PipelineRunCompleted)
    assert not any(isinstance(y, PipelineRunErrored) for y in yielded)

    # It carries the stopped reason + how much is left.
    payload = yielded[-1].payload
    assert payload["stopped_reason"] == "budget_exhausted"
    assert payload["documents_total"] == 3
    assert payload["documents_cognified"] == 1
    assert payload["documents_remaining"] == 2

    # Data is preserved: rollback must NOT run; the completed run is logged with the
    # same stopped-reason info persisted to run_info.
    assert rollback_calls == []
    assert (
        complete_calls
        and complete_calls[0]["run_info_extra"]["stopped_reason"] == "budget_exhausted"
    )
