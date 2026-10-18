"""``run_tasks(after_run_completed=...)``: a hook for work after a run is done.

Cognify uses it to compact the vector store (``compact_vector_store``). The
hook must run only for a completed run, only once the run is recorded and
reported complete, and it must never be able to turn a completed run into a
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
from cognee.infrastructure.databases.vector.compact_vector_store import compact_vector_store
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

    async def hook():
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
async def test_failing_hook_raises_without_failing_or_rolling_back_the_run(
    monkeypatch, runner_plumbing
):
    dataset, logs = _setup(monkeypatch, runner_plumbing, _ok_item)
    rollback = AsyncMock()

    async def hook():
        raise RuntimeError("compaction exploded")

    with pytest.raises(RuntimeError, match="compaction exploded"):
        await _drive(dataset, rollback_handler=rollback, after_run_completed=hook)

    assert logs.complete.await_count == 1
    assert logs.error.await_count == 0
    rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_cognify_compacts_the_vector_store_after_its_run():
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
    assert call["after_run_completed"] is compact_vector_store


def test_only_cognify_compacts():
    """add, memify, improve and custom pipelines do not compact: cognify is the
    pipeline that writes vectors in bulk, and every remember() runs it."""
    package_root = Path(cognee.__file__).parent
    referencing = set()
    for source_file in package_root.rglob("*.py"):
        if "tests" in source_file.relative_to(package_root).parts:
            continue
        source = source_file.read_text(encoding="utf-8")
        if "compact_vector_store" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name) and node.id == "compact_vector_store":
                referencing.add(source_file.relative_to(package_root).as_posix())
                break

    assert referencing == {"api/v1/cognify/cognify.py"}
