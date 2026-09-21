"""``run_tasks`` gathers items with ``return_exceptions=True`` (SDK-775).

One item's hard failure no longer abandons its in-flight siblings: gather()
waits for them, so each emits its own terminal telemetry, and the failure then
propagates exactly as before.
"""

import asyncio
import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

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


def _wire(monkeypatch, dataset, item_runner):
    async def _noop(*_args, **_kwargs):
        return None

    async def _log_start(*_args, **_kwargs):
        return SimpleNamespace(pipeline_run_id=uuid4())

    monkeypatch.setattr(run_tasks_module, "get_relational_engine", lambda: _FakeEngine(dataset))
    monkeypatch.setattr(run_tasks_module, "generate_pipeline_id", lambda *_args: uuid4())
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_start", _log_start)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", _noop)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_progress", _noop)
    monkeypatch.setattr(run_tasks_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", item_runner)


@pytest.mark.asyncio
async def test_hard_raising_item_waits_for_siblings_then_propagates(monkeypatch):
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    finished = []

    async def _item(data_item, *_args, **_kwargs):
        if data_item == "b":
            raise RuntimeError("hard failure")
        # Still in flight when "b" fails; before return_exceptions=True gather()
        # raised at that moment and left these running detached.
        await asyncio.sleep(0.05)
        finished.append(data_item)
        return {"run_info": "ok"}

    _wire(monkeypatch, dataset, _item)

    with pytest.raises(RuntimeError):
        async for _ in run_tasks_module.run_tasks(
            tasks=[],
            dataset_id=dataset.id,
            data=["a", "b", "c"],
            user=SimpleNamespace(id=uuid4(), tenant_id=None),
            pipeline_name="cognify_pipeline",
        ):
            pass

    assert sorted(finished) == ["a", "c"]
