"""Every dataset gets exactly ONE cognify_pipeline run.

_execute_cognify_runs maps planned invocations onto single run_pipeline calls:
a plain run for uniform datasets, an explicit-items run for manifest-only
datasets, and a multi-leg run for mixed datasets — never two runs (or two
pipeline names) for one dataset. run_tasks executes legs under a single run
lifecycle: one start record, per-leg task lists, one terminal status.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee.api.v1.cognify.cognify  # noqa: F401 — ensure the module (not the re-exported function) is importable via sys.modules
import cognee.modules.pipelines.operations.run_tasks as run_tasks_module
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunStarted,
)

# `from cognee.api.v1.cognify import cognify` would resolve to the re-exported
# cognify FUNCTION; grab the actual module object for patching.
cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]


class _ExecutorRecorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {ds: f"run_info_{ds}" for ds in kwargs["datasets"] or ["all"]}


async def _execute(plan, executor):
    return await cognify_module._execute_cognify_runs(
        plan,
        executor=executor,
        cognify_tasks="STANDARD_TASKS",
        chunk_size=None,
        chunks_per_batch=None,
        user=None,
        vector_db_config=None,
        graph_db_config=None,
        incremental_loading=True,
        data_per_batch=20,
        llm_config=None,
        embedding_config=None,
        data_cache=True,
    )


class TestExecuteCognifyRuns:
    @pytest.mark.asyncio
    async def test_standard_plan_is_one_plain_call(self):
        executor = _ExecutorRecorder()
        await _execute([{"datasets": ["a", "b"], "leg_kinds": None}], executor)

        (call,) = executor.calls
        assert call["pipeline_name"] == "cognify_pipeline"
        assert call["tasks"] == "STANDARD_TASKS"
        assert "data" not in call and "legs" not in call

    @pytest.mark.asyncio
    async def test_manifest_only_dataset_is_one_run_under_cognify_name(self):
        executor = _ExecutorRecorder()
        items = ["manifest_item"]
        with patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value="DLT_TASKS")):
            await _execute([{"datasets": ["ds1"], "leg_kinds": [("dlt", items)]}], executor)

        (call,) = executor.calls
        assert call["pipeline_name"] == "cognify_pipeline"  # not dlt_cognify_pipeline
        assert call["tasks"] == "DLT_TASKS"
        assert call["data"] == items
        assert "legs" not in call

    @pytest.mark.asyncio
    async def test_mixed_dataset_is_one_multi_leg_run(self):
        executor = _ExecutorRecorder()
        with patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value="DLT_TASKS")):
            await _execute(
                [
                    {
                        "datasets": ["ds1"],
                        "leg_kinds": [("dlt", ["m1"]), ("standard", ["r1", "r2"])],
                    }
                ],
                executor,
            )

        (call,) = executor.calls
        assert call["pipeline_name"] == "cognify_pipeline"
        assert call["legs"] == [("DLT_TASKS", ["m1"]), ("STANDARD_TASKS", ["r1", "r2"])]
        assert "tasks" not in call and "data" not in call

    @pytest.mark.asyncio
    async def test_results_union_across_disjoint_invocations(self):
        executor = _ExecutorRecorder()
        with patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value="DLT_TASKS")):
            result = await _execute(
                [
                    {"datasets": ["ds1"], "leg_kinds": [("dlt", ["m"])]},
                    {"datasets": ["ds2", "ds3"], "leg_kinds": None},
                ],
                executor,
            )

        assert result == {
            "ds1": "run_info_ds1",
            "ds2": "run_info_ds2",
            "ds3": "run_info_ds3",
        }


@pytest.mark.asyncio
async def test_run_tasks_legs_share_one_run_lifecycle(monkeypatch):
    dataset = SimpleNamespace(id=uuid4(), name="mixed_ds", owner_id=uuid4())
    run_id = uuid4()

    session = MagicMock()
    session.get = AsyncMock(return_value=dataset)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock(spec=["get_async_session"])  # no push_to_s3 attribute
    engine.get_async_session.return_value = session_ctx
    monkeypatch.setattr(run_tasks_module, "get_relational_engine", lambda: engine)

    db_ctx = MagicMock()
    db_ctx.return_value.__aenter__ = AsyncMock(return_value=None)
    db_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(run_tasks_module, "set_database_global_context_variables", db_ctx)

    log_start = AsyncMock(return_value=SimpleNamespace(pipeline_run_id=run_id))
    log_complete = AsyncMock()
    log_error = AsyncMock()
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_start", log_start)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_complete", log_complete)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", log_error)
    monkeypatch.setattr(
        run_tasks_module, "get_graph_engine", AsyncMock(return_value=SimpleNamespace())
    )

    processed = []

    async def _fake_item_run(data_item, ds, item_tasks, *args, **kwargs):
        processed.append((data_item, item_tasks))
        return {"run_info": "ok"}

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", _fake_item_run)

    events = []
    async for event in run_tasks_module.run_tasks.__wrapped__(
        tasks=None,
        dataset_id=dataset.id,
        data=None,
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="cognify_pipeline",
        legs=[("DLT_TASKS", ["m1", "m2"]), ("STANDARD_TASKS", ["r1"])],
    ):
        events.append(event)

    # One run record started and completed — never one per leg.
    assert log_start.await_count == 1
    assert log_complete.await_count == 1
    assert log_error.await_count == 0
    assert [type(e) for e in events] == [PipelineRunStarted, PipelineRunCompleted]
    assert all(e.pipeline_run_id == run_id for e in events)

    # Every item ran with its own leg's task list.
    assert sorted(processed) == [
        ("m1", "DLT_TASKS"),
        ("m2", "DLT_TASKS"),
        ("r1", "STANDARD_TASKS"),
    ]
