"""Guards the strong-reference anchor for the background pipeline task.

``run_pipeline_as_background_process`` launches the rest of the run with
``asyncio.create_task``. The event loop keeps only a *weak* reference to a
task, so unless the task is anchored somewhere it can be garbage-collected
mid-flight, silently aborting the background run. The fix anchors it in the
module-level ``_BACKGROUND_PIPELINE_TASKS`` set and discards it on completion.
"""

import gc
import asyncio

import pytest

from cognee.modules.pipelines.layers import pipeline_execution_mode
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
