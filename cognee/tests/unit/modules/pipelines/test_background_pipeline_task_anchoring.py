"""Guards the strong-reference anchor for the background pipeline task.

``run_pipeline_as_background_process`` launches the rest of the run with
``asyncio.create_task``. The event loop keeps only a *weak* reference to a
task, so unless the task is anchored somewhere it can be garbage-collected
mid-flight, silently aborting the background run. The fix anchors it in the
module-level ``_BACKGROUND_PIPELINE_TASKS`` set and discards it on completion.
"""

import gc
import asyncio
from uuid import uuid4

import pytest

from cognee.modules.pipelines.layers import pipeline_execution_mode
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
)
from cognee.modules.pipelines.layers.pipeline_execution_mode import (
    run_pipeline_as_background_process,
)


class _FakeRunInfo:
    """Minimal stand-in for the run-info objects the pipeline yields."""

    def __init__(self, dataset_id, payload=None, pipeline_run_id="run-1"):
        self.dataset_id = dataset_id
        self.payload = payload
        self.pipeline_run_id = pipeline_run_id


@pytest.mark.asyncio
async def test_background_pipeline_task_is_anchored_until_done(monkeypatch):
    # The background task pushes run info to a queue; stub it out so the test
    # exercises only the task lifecycle.
    monkeypatch.setattr(pipeline_execution_mode, "push_to_queue", lambda *a, **k: None)

    release = asyncio.Event()

    async def pipeline(**kwargs):
        # First item is consumed synchronously to build the "started" response.
        yield _FakeRunInfo(dataset_id="ds1", payload=["payload"])
        # Block the background task mid-run so the anchor can be observed.
        await release.wait()
        yield _FakeRunInfo(dataset_id="ds1", pipeline_run_id="run-1")

    started = await run_pipeline_as_background_process(pipeline, datasets="ds1")
    assert "ds1" in started

    # While the task is in flight it must be strongly referenced, otherwise the
    # event loop's weak reference lets gc collect it before it finishes.
    assert len(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS) == 1
    task = next(iter(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS))

    # The anchor must survive a garbage collection while the task runs.
    gc.collect()
    assert task in pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS

    # Let the task finish; the done-callback must remove it from the anchor set.
    release.set()
    await task
    await asyncio.sleep(0)  # allow the done-callback to run
    assert task not in pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS
    assert len(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS) == 0


@pytest.mark.asyncio
async def test_background_failure_does_not_abort_later_datasets(monkeypatch):
    first_dataset_id = uuid4()
    second_dataset_id = uuid4()
    run_ids = {
        first_dataset_id: uuid4(),
        second_dataset_id: uuid4(),
    }
    queued = []
    persisted_errors = []

    async def record_error(*args):
        persisted_errors.append(args)

    monkeypatch.setattr(
        pipeline_execution_mode,
        "push_to_queue",
        lambda pipeline_run_id, run_info: queued.append((pipeline_run_id, run_info)),
    )
    monkeypatch.setattr(
        pipeline_execution_mode,
        "_record_unhandled_pipeline_error",
        record_error,
    )

    async def pipeline(datasets, **_kwargs):
        run_id = run_ids[datasets]
        yield PipelineRunStarted(
            pipeline_run_id=run_id,
            dataset_id=datasets,
            dataset_name=str(datasets),
        )
        if datasets == first_dataset_id:
            yield PipelineRunErrored(
                pipeline_run_id=run_id,
                dataset_id=datasets,
                dataset_name=str(datasets),
                payload="failed",
            )
            raise RuntimeError("first dataset failed")
        yield PipelineRunCompleted(
            pipeline_run_id=run_id,
            dataset_id=datasets,
            dataset_name=str(datasets),
        )

    started = await run_pipeline_as_background_process(
        pipeline,
        datasets=[first_dataset_id, second_dataset_id],
    )
    assert set(started) == {first_dataset_id, second_dataset_id}

    task = next(iter(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS))
    await task
    await asyncio.sleep(0)

    first_updates = [info for _, info in queued if info.dataset_id == first_dataset_id]
    second_updates = [info for _, info in queued if info.dataset_id == second_dataset_id]
    assert [type(info) for info in first_updates] == [PipelineRunErrored]
    assert [type(info) for info in second_updates] == [PipelineRunCompleted]
    assert persisted_errors == []
    assert task not in pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS


@pytest.mark.asyncio
async def test_background_supervisor_synthesizes_error_and_closes_generators(monkeypatch):
    first_dataset_id = uuid4()
    second_dataset_id = uuid4()
    run_ids = {
        first_dataset_id: uuid4(),
        second_dataset_id: uuid4(),
    }
    queued = []
    closed = []
    persisted_errors = []

    async def record_error(started_info, error):
        persisted_errors.append((started_info, error))

    monkeypatch.setattr(
        pipeline_execution_mode,
        "push_to_queue",
        lambda pipeline_run_id, run_info: queued.append((pipeline_run_id, run_info)),
    )
    monkeypatch.setattr(
        pipeline_execution_mode,
        "_record_unhandled_pipeline_error",
        record_error,
    )

    async def pipeline(datasets, **_kwargs):
        try:
            run_id = run_ids[datasets]
            yield PipelineRunStarted(
                pipeline_run_id=run_id,
                dataset_id=datasets,
                dataset_name=str(datasets),
            )
            if datasets == first_dataset_id:
                raise RuntimeError("failed without a terminal update")
            yield PipelineRunCompleted(
                pipeline_run_id=run_id,
                dataset_id=datasets,
                dataset_name=str(datasets),
            )
        finally:
            closed.append(datasets)

    await run_pipeline_as_background_process(
        pipeline,
        datasets=[first_dataset_id, second_dataset_id],
    )
    task = next(iter(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS))
    await task
    await asyncio.sleep(0)

    first_updates = [info for _, info in queued if info.dataset_id == first_dataset_id]
    second_updates = [info for _, info in queued if info.dataset_id == second_dataset_id]
    assert len(first_updates) == 1
    assert isinstance(first_updates[0], PipelineRunErrored)
    assert "failed without a terminal update" in first_updates[0].payload
    assert [type(info) for info in second_updates] == [PipelineRunCompleted]
    assert set(closed) == {first_dataset_id, second_dataset_id}
    assert len(persisted_errors) == 1
    assert persisted_errors[0][0].dataset_id == first_dataset_id
    assert str(persisted_errors[0][1]) == "failed without a terminal update"


def test_background_done_callback_retrieves_and_logs_unexpected_exception(monkeypatch):
    error = RuntimeError("supervisor failed")

    class _FailedTask:
        exception_calls = 0

        def cancelled(self):
            return False

        def exception(self):
            self.exception_calls += 1
            return error

    failed_task = _FailedTask()
    logged = []
    monkeypatch.setattr(
        pipeline_execution_mode.logger,
        "error",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )
    pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS.add(failed_task)

    pipeline_execution_mode._handle_background_task_done(failed_task)

    assert failed_task.exception_calls == 1
    assert failed_task not in pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS
    assert logged


@pytest.mark.asyncio
async def test_background_cancellation_closes_all_started_generators(monkeypatch):
    first_dataset_id = uuid4()
    second_dataset_id = uuid4()
    release = asyncio.Event()
    closed = []

    monkeypatch.setattr(pipeline_execution_mode, "push_to_queue", lambda *args: None)

    async def pipeline(datasets, **_kwargs):
        try:
            yield PipelineRunStarted(
                pipeline_run_id=uuid4(),
                dataset_id=datasets,
                dataset_name=str(datasets),
            )
            await release.wait()
        finally:
            closed.append(datasets)

    await run_pipeline_as_background_process(
        pipeline,
        datasets=[first_dataset_id, second_dataset_id],
    )
    task = next(iter(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS))
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert set(closed) == {first_dataset_id, second_dataset_id}
    assert task not in pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS
