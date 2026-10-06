"""``run_tasks(after_run_completed=...)``: a hook for work after a run is done.

Cognify passes the maintenance runner (``run_maintenance``). The hook must
run only for a completed run, only once the run is recorded and reported
complete, and it must never be able to turn a completed run into a
failed or rolled-back one.
"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee
import cognee.api.v1.cognify.cognify  # registers the module in sys.modules
import cognee.modules.pipelines.operations.run_tasks as run_tasks_module
from cognee.modules.maintenance import run_maintenance
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted

cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]


def _setup(monkeypatch, runner_plumbing, item_run):
    dataset = SimpleNamespace(id=uuid4(), name="hook_ds", owner_id=uuid4())
    logs = runner_plumbing(run_tasks_module, dataset)
    monkeypatch.setattr(run_tasks_module, "validate_pipeline_tasks", lambda tasks: None)
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", item_run)
    return dataset, logs


async def _drive(dataset, rollback_handler=None, **kwargs):
    events = []
    async for event in run_tasks_module.run_tasks(
        tasks=["TASKS"],
        dataset_id=dataset.id,
        data=["item"],
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="cognify_pipeline",
        rollback_handler=rollback_handler,
        **kwargs,
    ):
        events.append(event)
    return events


async def _ok_item(*args, **kwargs):
    return {"run_info": "ok"}


@pytest.mark.asyncio
async def test_hook_runs_once_after_the_run_is_reported_complete(monkeypatch, runner_plumbing):
    dataset, logs = _setup(monkeypatch, runner_plumbing, _ok_item)
    order = []
    logs.complete.side_effect = lambda *a, **k: order.append("logged_complete")

    received = {}

    async def hook(**kwargs):
        received.update(kwargs)
        order.append("hook")

    events = []
    async for event in run_tasks_module.run_tasks(
        tasks=["TASKS"],
        dataset_id=dataset.id,
        data=["item"],
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="cognify_pipeline",
        after_run_completed=hook,
    ):
        if isinstance(event, PipelineRunCompleted):
            order.append("yielded_complete")
        events.append(event)

    assert order == ["logged_complete", "yielded_complete", "hook"]
    assert logs.error.await_count == 0
    # The same keyword arguments rollback_handler gets.
    assert received["pipeline_name"] == "cognify_pipeline"
    assert received["dataset"] is dataset
    assert received["pipeline_run_id"] == logs.start.return_value.pipeline_run_id
    assert set(received) >= {"pipeline_id", "user", "data", "data_ingestion_info"}


@pytest.mark.asyncio
async def test_hook_is_not_called_for_a_failed_run(monkeypatch, runner_plumbing):
    async def failing_item(*args, **kwargs):
        raise RuntimeError("extraction failed")

    dataset, logs = _setup(monkeypatch, runner_plumbing, failing_item)
    hook = AsyncMock()

    with pytest.raises(RuntimeError):
        await _drive(dataset, after_run_completed=hook)

    assert logs.error.await_count == 1
    hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_failing_hook_cannot_fail_or_roll_back_a_completed_run(monkeypatch, runner_plumbing):
    dataset, logs = _setup(monkeypatch, runner_plumbing, _ok_item)
    rollback = AsyncMock()

    async def hook(**kwargs):
        raise RuntimeError("compaction exploded")

    events = await _drive(dataset, rollback_handler=rollback, after_run_completed=hook)

    assert isinstance(events[-1], PipelineRunCompleted)
    assert logs.complete.await_count == 1
    assert logs.error.await_count == 0
    rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_cognify_runs_maintenance_after_its_run():
    calls = []

    async def fake_executor(**kwargs):
        calls.append(kwargs)
        return {}

    with (
        patch.object(
            cognify_module, "get_pipeline_executor", lambda run_in_background: fake_executor
        ),
        patch.object(cognify_module, "get_default_tasks", new=AsyncMock(return_value=[])),
        patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value=[])),
        patch.object(cognify_module, "get_code_file_tasks", new=MagicMock(return_value=[])),
        patch.object(cognify_module, "get_code_repo_tasks", new=MagicMock(return_value=[])),
    ):
        await cognify_module.cognify(
            datasets=["ds"],
            chunk_size=1024,
            config={"ontology_config": {"ontology_resolver": None}},
        )

    (call,) = calls
    assert call["after_run_completed"] is run_maintenance


def test_only_cognify_runs_maintenance():
    """add, memify, improve and custom pipelines do not opt in: cognify is the
    pipeline that writes vectors in bulk, and every remember() runs it. Each
    job's ``pipelines`` then picks which jobs follow it."""
    package_root = Path(cognee.__file__).parent
    referencing = set()
    for source_file in package_root.rglob("*.py"):
        relative = source_file.relative_to(package_root)
        if "tests" in relative.parts or relative.parts[:2] == ("modules", "maintenance"):
            continue
        source = source_file.read_text(encoding="utf-8")
        if "run_maintenance" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name) and node.id == "run_maintenance":
                referencing.add(relative.as_posix())
                break

    assert referencing == {"api/v1/cognify/cognify.py"}
