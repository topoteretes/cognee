"""``run_tasks(after_run_completed=..., after_items_ran=...)``: hooks for work after a run is done.

Cognify uses ``after_run_completed`` to compact the vector store
(``compact_vector_store``). That hook must run only for a completed run, only
once the run is recorded and reported complete, and it must never be able to
turn a completed run into a failed or rolled-back one. ``after_items_ran`` (cognify
reconciles the document structure with it) also runs when items in the run
failed, because one failing item must not freeze the rest.
"""

import ast
import asyncio
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
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunErrored

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
async def test_cognify_reconciles_structure_and_compacts_after_its_run():
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
    assert call["after_items_ran"] is cognify_module.reconcile_document_structure


async def _reported_failure_item(*args, **kwargs):
    """An item that reports its failure as a result (RAISE_INCREMENTAL_LOADING_ERRORS=false)
    instead of raising it."""
    return {
        "run_info": PipelineRunErrored(
            pipeline_run_id=uuid4(), payload="LLM refused", dataset_id=uuid4(), dataset_name="d"
        ),
        "error": RuntimeError("LLM refused"),
    }


@pytest.mark.asyncio
async def test_after_items_ran_still_runs_when_an_item_failed_but_after_run_completed_does_not(
    monkeypatch, runner_plumbing
):
    dataset, logs = _setup(monkeypatch, runner_plumbing, _reported_failure_item)
    after_items_ran, after_run_completed = AsyncMock(), AsyncMock()

    events = await _drive(
        dataset, after_items_ran=after_items_ran, after_run_completed=after_run_completed
    )

    assert any(isinstance(event, PipelineRunErrored) for event in events)
    assert logs.error.await_count == 1
    after_items_ran.assert_awaited_once()
    after_run_completed.assert_not_awaited()


@pytest.mark.asyncio
async def test_after_items_ran_runs_when_an_item_raises_its_own_error(monkeypatch, runner_plumbing):
    """The default (RAISE_INCREMENTAL_LOADING_ERRORS=true): a failing item re-raises its own
    exception instead of reporting a failure, and the run ends with that error."""

    async def raising_item(*args, **kwargs):
        raise RuntimeError("extraction failed")

    dataset, logs = _setup(monkeypatch, runner_plumbing, raising_item)
    rollback, after_items_ran, after_run_completed = AsyncMock(), AsyncMock(), AsyncMock()

    with pytest.raises(RuntimeError, match="extraction failed"):
        await _drive(
            dataset,
            rollback_handler=rollback,
            after_items_ran=after_items_ran,
            after_run_completed=after_run_completed,
        )

    # Rolled back and recorded as errored first, then the hook, then the error goes on.
    rollback.assert_awaited_once()
    assert logs.error.await_count == 1
    after_items_ran.assert_awaited_once()
    after_run_completed.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failing_after_items_ran_on_a_failed_item_keeps_the_item_error(
    monkeypatch, runner_plumbing
):
    """The run's own error is what callers map to a status (an exhausted LLM budget is a
    402), so a broken hook must not replace it. The hook's error stays as its cause, and
    the run must already be rolled back and recorded."""

    async def raising_item(*args, **kwargs):
        raise RuntimeError("extraction failed")

    async def broken_hook():
        raise ValueError("structure pass failed")

    dataset, logs = _setup(monkeypatch, runner_plumbing, raising_item)
    rollback = AsyncMock()

    with pytest.raises(RuntimeError, match="extraction failed") as raised:
        await _drive(dataset, rollback_handler=rollback, after_items_ran=broken_hook)

    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == "structure pass failed"
    rollback.assert_awaited_once()
    assert logs.error.await_count == 1


@pytest.mark.asyncio
async def test_after_items_ran_does_not_run_when_the_run_is_cancelled(monkeypatch, runner_plumbing):
    async def cancelled_item(*args, **kwargs):
        raise asyncio.CancelledError

    dataset, _logs = _setup(monkeypatch, runner_plumbing, cancelled_item)
    after_items_ran = AsyncMock()

    with pytest.raises(asyncio.CancelledError):
        await _drive(dataset, after_items_ran=after_items_ran)

    after_items_ran.assert_not_awaited()


@pytest.mark.parametrize("name", ["compact_vector_store", "reconcile_document_structure"])
def test_only_cognify_runs_the_after_hooks(name):
    """add, memify, improve and custom pipelines run neither: cognify is the pipeline
    that writes vectors in bulk, and every remember() runs it. The structure pass
    deletes edges over a whole dataset, so a second caller would repeat that."""
    package_root = Path(cognee.__file__).parent
    referencing = set()
    for source_file in package_root.rglob("*.py"):
        relative = source_file.relative_to(package_root)
        if "tests" in relative.parts:
            continue
        source = source_file.read_text(encoding="utf-8")
        if name not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name) and node.id == name:
                referencing.add(relative.as_posix())
                break

    assert referencing == {"api/v1/cognify/cognify.py"}
