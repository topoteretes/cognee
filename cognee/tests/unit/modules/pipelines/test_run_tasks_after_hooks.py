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
import cognee.modules.pipelines.operations.pipeline as pipeline_module
import cognee.modules.pipelines.operations.run_tasks as run_tasks_module
from cognee.infrastructure.databases.vector.compact_vector_store import compact_vector_store
from cognee.modules.pipelines.exceptions import AfterRunHookError
from cognee.modules.pipelines.layers import pipeline_execution_mode
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

    with pytest.raises(AfterRunHookError) as raised:
        await _drive(dataset, rollback_handler=rollback, after_run_completed=hook)

    assert str(raised.value.__cause__) == "compaction exploded"
    assert raised.value.failures == [(dataset.id, "hook_ds", raised.value.__cause__)]
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
async def test_a_failing_after_items_ran_after_a_reported_failure_keeps_the_reported_error(
    monkeypatch, runner_plumbing
):
    """A reported failure is returned, not raised, and callers act on its error class
    (improve stops its run on a budget error). A broken hook must not replace it."""
    from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError

    async def budget_item(*args, **kwargs):
        return {
            "run_info": PipelineRunErrored(
                pipeline_run_id=uuid4(), payload="budget", dataset_id=uuid4(), dataset_name="d"
            ),
            "error": LLMPaymentRequiredError(),
        }

    async def broken_hook():
        raise ValueError("structure pass failed")

    dataset, _logs = _setup(monkeypatch, runner_plumbing, budget_item)

    events = await _drive(dataset, after_items_ran=broken_hook)

    errored = [event for event in events if isinstance(event, PipelineRunErrored)]
    assert [event.error_class for event in errored] == ["LLMPaymentRequiredError"]


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


def _hook_failure(dataset, message="compaction exploded"):
    return AfterRunHookError([(dataset.id, dataset.name, RuntimeError(message))])


@pytest.mark.asyncio
async def test_a_failed_hook_does_not_stop_the_next_datasets(monkeypatch):
    """Each dataset's run is already complete when its hook fails, so the remaining
    datasets still run, and every failure is raised together after them."""
    datasets = [SimpleNamespace(id=uuid4(), name=name) for name in ("a", "b", "c")]
    ran = []

    async def run_pipeline_per_dataset(dataset, **kwargs):
        ran.append(dataset.name)
        yield PipelineRunCompleted(
            pipeline_run_id=uuid4(), dataset_id=dataset.id, dataset_name=dataset.name
        )
        if dataset.name != "b":
            raise _hook_failure(dataset)

    monkeypatch.setattr(pipeline_module, "setup_and_check_environment", AsyncMock())
    monkeypatch.setattr(
        pipeline_module,
        "resolve_authorized_user_datasets",
        AsyncMock(return_value=(SimpleNamespace(id=uuid4()), datasets)),
    )
    monkeypatch.setattr(pipeline_module, "run_pipeline_per_dataset", run_pipeline_per_dataset)

    events = []
    with pytest.raises(AfterRunHookError) as raised:
        async for event in pipeline_module.run_pipeline(tasks=lambda item: []):
            events.append(event)

    assert ran == ["a", "b", "c"]
    assert len(events) == 3
    assert [name for _id, name, _error in raised.value.failures] == ["a", "c"]
    assert raised.value.__cause__ is raised.value.failures[0][2]
    assert "'a' (RuntimeError: compaction exploded)" in str(raised.value)


@pytest.mark.asyncio
async def test_a_task_that_raises_a_hook_error_fails_its_run(monkeypatch, runner_plumbing):
    """An improve bridge raises the hook error of the cognify it ran; that fails the
    bridge's own run and goes on unchanged, not as a failed step after it."""
    failure = AfterRunHookError([(uuid4(), "inner", "RuntimeError: compaction")])

    async def raising_item(*args, **kwargs):
        raise failure

    dataset, _logs = _setup(monkeypatch, runner_plumbing, raising_item)

    def run_pipeline_per_dataset(dataset, **kwargs):
        return run_tasks_module.run_tasks(
            tasks=["TASKS"],
            dataset_id=dataset.id,
            data=["item"],
            user=SimpleNamespace(id=uuid4(), tenant_id=None),
            pipeline_name="memify_pipeline",
        )

    monkeypatch.setattr(pipeline_module, "setup_and_check_environment", AsyncMock())
    monkeypatch.setattr(
        pipeline_module,
        "resolve_authorized_user_datasets",
        AsyncMock(return_value=(SimpleNamespace(id=uuid4()), [dataset])),
    )
    monkeypatch.setattr(pipeline_module, "run_pipeline_per_dataset", run_pipeline_per_dataset)

    events = []
    with pytest.raises(AfterRunHookError) as raised:
        async for event in pipeline_module.run_pipeline(tasks=lambda item: []):
            events.append(event)

    assert raised.value is failure
    assert isinstance(events[-1], PipelineRunErrored)


@pytest.mark.asyncio
async def test_failed_steps_are_chained_from_the_first_exception(monkeypatch):
    """A failure can carry a reason string instead of an exception; the raise
    chains from the first real exception."""
    datasets = [SimpleNamespace(id=uuid4(), name=name) for name in ("a", "b")]
    compaction = RuntimeError("compaction exploded")
    failures = {"a": "RuntimeError: earlier", "b": compaction}

    async def run_pipeline_per_dataset(dataset, **kwargs):
        yield PipelineRunCompleted(
            pipeline_run_id=uuid4(), dataset_id=dataset.id, dataset_name=dataset.name
        )
        raise AfterRunHookError([(dataset.id, dataset.name, failures[dataset.name])])

    monkeypatch.setattr(pipeline_module, "setup_and_check_environment", AsyncMock())
    monkeypatch.setattr(
        pipeline_module,
        "resolve_authorized_user_datasets",
        AsyncMock(return_value=(SimpleNamespace(id=uuid4()), datasets)),
    )
    monkeypatch.setattr(pipeline_module, "run_pipeline_per_dataset", run_pipeline_per_dataset)

    with pytest.raises(AfterRunHookError) as raised:
        async for _event in pipeline_module.run_pipeline(tasks=lambda item: []):
            pass

    assert [name for _id, name, _error in raised.value.failures] == ["a", "b"]
    assert raised.value.__cause__ is compaction


@pytest.mark.asyncio
async def test_the_blocking_executor_marks_each_run_whose_hook_failed():
    """The raised error carries every dataset's run, and only the runs whose step
    failed carry the reason, so a caller that does not raise can still tell them apart."""
    datasets = [SimpleNamespace(id=uuid4(), name=name) for name in ("a", "b")]

    async def pipeline():
        for dataset in datasets:
            yield PipelineRunCompleted(
                pipeline_run_id=uuid4(), dataset_id=dataset.id, dataset_name=dataset.name
            )
        raise _hook_failure(datasets[0])

    with pytest.raises(AfterRunHookError) as raised:
        await pipeline_execution_mode.run_pipeline_blocking(pipeline)

    runs = raised.value.run_info
    assert runs[datasets[0].id].after_run_error == "RuntimeError: compaction exploded"
    assert runs[datasets[1].id].after_run_error is None
    assert "after_run_error" not in runs[datasets[1].id].model_dump()


@pytest.mark.asyncio
async def test_a_failed_hook_in_the_background_does_not_stop_the_next_datasets(monkeypatch):
    pushed = []
    monkeypatch.setattr(
        pipeline_execution_mode, "push_to_queue", lambda run_id, info: pushed.append(info)
    )

    async def pipeline(datasets):
        dataset = SimpleNamespace(id=datasets, name=datasets)
        yield SimpleNamespace(dataset_id=datasets, payload=None, pipeline_run_id=datasets)
        yield SimpleNamespace(dataset_id=datasets, payload="done", pipeline_run_id=datasets)
        if datasets == "first":
            raise _hook_failure(dataset)

    await pipeline_execution_mode.run_pipeline_as_background_process(
        pipeline, datasets=["first", "second"]
    )
    (task,) = pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS
    await task

    assert [(info.dataset_id, info.payload) for info in pushed] == [
        ("first", "done"),
        ("second", "done"),
    ]


@pytest.mark.asyncio
async def test_cognify_raises_a_failed_hook_unwrapped():
    """Not a CognifyFailedError: the runs completed, and the error says so."""
    failure = _hook_failure(SimpleNamespace(id=uuid4(), name="ds"))

    async def fake_executor(**kwargs):
        raise failure

    with (
        patch.object(
            cognify_module, "get_pipeline_executor", lambda run_in_background: fake_executor
        ),
        patch.object(cognify_module, "get_default_tasks", new=AsyncMock(return_value=[])),
        patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value=[])),
        patch.object(cognify_module, "get_code_file_tasks", new=MagicMock(return_value=[])),
        patch.object(cognify_module, "get_code_repo_tasks", new=MagicMock(return_value=[])),
        pytest.raises(AfterRunHookError) as raised,
    ):
        await cognify_module.cognify(
            datasets=["ds"],
            chunk_size=1024,
            config={"ontology_config": {"ontology_resolver": None}},
        )

    assert raised.value is failure


@pytest.mark.asyncio
async def test_cognify_without_raise_on_error_returns_the_runs_of_a_failed_hook():
    """raise_on_error=False reports a failed step on the runs, as it does an errored run."""
    dataset = SimpleNamespace(id=uuid4(), name="ds")
    completed = PipelineRunCompleted(
        pipeline_run_id=uuid4(),
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        after_run_error="RuntimeError: compaction exploded",
    )
    failure = _hook_failure(dataset)
    failure.run_info = {dataset.id: completed}

    async def fake_executor(**kwargs):
        raise failure

    with (
        patch.object(
            cognify_module, "get_pipeline_executor", lambda run_in_background: fake_executor
        ),
        patch.object(cognify_module, "get_default_tasks", new=AsyncMock(return_value=[])),
        patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value=[])),
        patch.object(cognify_module, "get_code_file_tasks", new=MagicMock(return_value=[])),
        patch.object(cognify_module, "get_code_repo_tasks", new=MagicMock(return_value=[])),
    ):
        result = await cognify_module.cognify(
            datasets=["ds"],
            chunk_size=1024,
            config={"ontology_config": {"ontology_resolver": None}},
            raise_on_error=False,
        )

    assert result == {dataset.id: completed}


@pytest.mark.asyncio
async def test_cognify_raises_an_errored_dataset_before_a_failed_hook():
    """One dataset's run errored, another's step after its run failed: the errored
    run is raised, not hidden behind an error that says the data is stored."""
    from cognee.modules.pipelines.exceptions import CognifyFailedError

    errored_ds = SimpleNamespace(id=uuid4(), name="a")
    hook_ds = SimpleNamespace(id=uuid4(), name="b")
    failure = _hook_failure(hook_ds)
    failure.run_info = {
        errored_ds.id: PipelineRunErrored(
            pipeline_run_id=uuid4(),
            dataset_id=errored_ds.id,
            dataset_name=errored_ds.name,
            error_class="ValueError",
            error_message="item failed",
        ),
        hook_ds.id: PipelineRunCompleted(
            pipeline_run_id=uuid4(), dataset_id=hook_ds.id, dataset_name=hook_ds.name
        ),
    }

    async def fake_executor(**kwargs):
        raise failure

    with (
        patch.object(
            cognify_module, "get_pipeline_executor", lambda run_in_background: fake_executor
        ),
        patch.object(cognify_module, "get_default_tasks", new=AsyncMock(return_value=[])),
        patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value=[])),
        patch.object(cognify_module, "get_code_file_tasks", new=MagicMock(return_value=[])),
        patch.object(cognify_module, "get_code_repo_tasks", new=MagicMock(return_value=[])),
        pytest.raises(CognifyFailedError, match="item failed") as raised,
    ):
        await cognify_module.cognify(
            datasets=["a", "b"],
            chunk_size=1024,
            config={"ontology_config": {"ontology_resolver": None}},
        )

    assert raised.value.__context__ is failure


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
            # A call, a module attribute or an import alias are all a use of the hook.
            if (
                (isinstance(node, ast.Name) and node.id == name)
                or (isinstance(node, ast.Attribute) and node.attr == name)
                or (isinstance(node, ast.alias) and node.name == name)
            ):
                referencing.add(relative.as_posix())
                break

    assert referencing == {"api/v1/cognify/cognify.py"}
