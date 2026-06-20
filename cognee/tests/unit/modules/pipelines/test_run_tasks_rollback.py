from contextlib import asynccontextmanager
import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

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
async def test_run_tasks_calls_custom_rollback_on_pipeline_failure(monkeypatch):
    dataset_id = uuid4()
    user_id = uuid4()
    owner_id = uuid4()
    pipeline_run_id = uuid4()

    dataset = SimpleNamespace(id=dataset_id, name="dataset-1", owner_id=owner_id)
    user = SimpleNamespace(id=user_id, tenant_id=uuid4())
    data_item = SimpleNamespace(id=uuid4())

    async def _failing_item(*_args, **_kwargs):
        return {
            "run_info": PipelineRunErrored(
                pipeline_run_id=pipeline_run_id,
                dataset_id=dataset_id,
                dataset_name=dataset.name,
                payload="item failed",
            )
        }

    rollback_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    monkeypatch.setattr(run_tasks_module, "get_relational_engine", lambda: _FakeEngine(dataset))
    monkeypatch.setattr(run_tasks_module, "generate_pipeline_id", lambda *_args: uuid4())

    async def _log_start(*_args, **_kwargs):
        return SimpleNamespace(pipeline_run_id=pipeline_run_id)

    async def _log_error(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        run_tasks_module,
        "log_pipeline_run_start",
        _log_start,
    )
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", _log_error)
    monkeypatch.setattr(run_tasks_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", _failing_item)

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

    assert len(yielded) == 2
    assert isinstance(yielded[0], PipelineRunStarted)
    assert isinstance(yielded[1], PipelineRunErrored)

    assert len(rollback_calls) == 1
    rollback_payload = rollback_calls[0]
    assert rollback_payload["pipeline_run_id"] == pipeline_run_id
    assert rollback_payload["dataset"] == dataset
    assert rollback_payload["user"] == user
    assert rollback_payload["data"] == [data_item]
    assert isinstance(rollback_payload["error"], Exception)
    assert rollback_payload["data_ingestion_info"][0]["run_info"].status == "PipelineRunErrored"


@pytest.mark.asyncio
async def test_run_tasks_converts_unhandled_item_exceptions_to_run_errors(monkeypatch):
    dataset_id = uuid4()
    user_id = uuid4()
    owner_id = uuid4()
    pipeline_run_id = uuid4()

    dataset = SimpleNamespace(id=dataset_id, name="dataset-1", owner_id=owner_id)
    user = SimpleNamespace(id=user_id, tenant_id=uuid4())
    data_item = SimpleNamespace(id=uuid4())

    async def _raising_item(*_args, **_kwargs):
        raise RuntimeError("item exploded")

    rollback_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    monkeypatch.setattr(run_tasks_module, "get_relational_engine", lambda: _FakeEngine(dataset))
    monkeypatch.setattr(run_tasks_module, "generate_pipeline_id", lambda *_args: uuid4())

    async def _log_start(*_args, **_kwargs):
        return SimpleNamespace(pipeline_run_id=pipeline_run_id)

    async def _log_error(*_args, **_kwargs):
        return None

    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_start", _log_start)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", _log_error)
    monkeypatch.setattr(run_tasks_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", _raising_item)

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

    assert len(yielded) == 2
    assert isinstance(yielded[0], PipelineRunStarted)
    assert isinstance(yielded[1], PipelineRunErrored)
    assert yielded[1].data_ingestion_info[0]["run_info"].payload == "RuntimeError('item exploded')"
    assert rollback_calls[0]["data_ingestion_info"][0]["run_info"].payload == (
        "RuntimeError('item exploded')"
    )
