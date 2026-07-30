"""The distributed runner resolves per-item task lists in the orchestrator.

Modal workers must receive plain task lists (no callable is serialized), so
resolve_tasks is applied while building the dispatch columns: the task column
is [resolve_tasks(item) for item in data] instead of the [tasks] * n
broadcast. Mixed-kind datasets therefore RUN distributed — the old
partition-era NotImplementedError is gone.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.run_tasks_distributed as dist_module


class _FakeModalFn:
    """Stands in for run_tasks_on_modal; captures the dispatch columns."""

    def __init__(self):
        self.captured_columns = None
        parent = self

        class _Map:
            @staticmethod
            async def aio(*columns):
                parent.captured_columns = columns
                for _ in range(len(columns[0])):
                    yield {"run_info": "ok"}

        self.map = _Map()


@pytest.mark.asyncio
async def test_task_column_is_resolved_per_item(monkeypatch):
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())

    session = MagicMock()
    session.get = AsyncMock(return_value=dataset)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock(spec=["get_async_session"])
    engine.get_async_session.return_value = session_ctx
    monkeypatch.setattr(dist_module, "get_relational_engine", lambda: engine)

    db_ctx = MagicMock()
    db_ctx.return_value.__aenter__ = AsyncMock(return_value=None)
    db_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(dist_module, "set_database_global_context_variables", db_ctx)

    monkeypatch.setattr(
        dist_module,
        "log_pipeline_run_start",
        AsyncMock(return_value=SimpleNamespace(pipeline_run_id=uuid4())),
    )
    monkeypatch.setattr(dist_module, "log_pipeline_run_complete", AsyncMock())
    monkeypatch.setattr(dist_module, "log_pipeline_run_error", AsyncMock())

    async def _identity(data):
        return data

    monkeypatch.setattr(dist_module, "resolve_data_directories", _identity)

    validated = []
    monkeypatch.setattr(dist_module, "validate_pipeline_tasks", validated.append)

    fake_modal = _FakeModalFn()
    monkeypatch.setattr(dist_module, "run_tasks_on_modal", fake_modal, raising=False)

    dlt_list, std_list = ["DLT"], ["STD"]
    events = []
    async for event in dist_module.run_tasks_distributed(
        tasks=std_list,
        dataset_id=dataset.id,
        data=["m1", "r1", "m2"],
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="cognify_pipeline",
        resolve_tasks=lambda item: dlt_list if item.startswith("m") else std_list,
    ):
        events.append(event)

    # The task column carries each item's OWN resolved list, in item order.
    task_column = fake_modal.captured_columns[2]
    assert task_column == [dlt_list, std_list, dlt_list]
    # Each distinct resolved list validated exactly once.
    assert len(validated) == 2

    # Without a resolver, the broadcast shape is preserved.
    fake_modal_plain = _FakeModalFn()
    monkeypatch.setattr(dist_module, "run_tasks_on_modal", fake_modal_plain, raising=False)
    async for _ in dist_module.run_tasks_distributed(
        tasks=std_list,
        dataset_id=dataset.id,
        data=["a", "b"],
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="cognify_pipeline",
    ):
        pass
    assert fake_modal_plain.captured_columns[2] == [std_list, std_list]
