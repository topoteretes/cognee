"""Guards the strong-reference anchor for the background sync task.

``sync`` launches ``_perform_background_sync`` with ``asyncio.create_task`` and
returns immediately. The event loop keeps only a *weak* reference to a task, so
unless the task is anchored somewhere it can be garbage-collected mid-flight,
silently aborting the sync. The fix anchors it in the module-level
``_BACKGROUND_SYNC_TASKS`` set and discards it on completion.
"""

import gc
import asyncio
import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

# The ``cognee.api.v1.sync`` package re-exports the ``sync`` function, shadowing
# the submodule attribute — load the submodule explicitly so monkeypatching the
# module-level globals works.
sync_module = importlib.import_module("cognee.api.v1.sync.sync")


@pytest.mark.asyncio
async def test_background_sync_task_is_anchored_until_done(monkeypatch):
    # Avoid a real DB write for the sync-operation record.
    async def _noop_create(*args, **kwargs):
        return None

    monkeypatch.setattr(sync_module, "create_sync_operation", _noop_create)

    release = asyncio.Event()

    # Replace the heavy background routine with a controllable coroutine so the
    # test exercises only the task lifecycle.
    async def fake_background_sync(run_id, datasets, user):
        await release.wait()

    monkeypatch.setattr(sync_module, "_perform_background_sync", fake_background_sync)

    datasets = [SimpleNamespace(id=uuid4(), name="ds1")]
    user = SimpleNamespace(id=uuid4())

    response = await sync_module.sync(datasets, user)
    assert response.status == "started"

    # While the task is in flight it must be strongly referenced, otherwise the
    # event loop's weak reference lets gc collect it before it finishes.
    assert len(sync_module._BACKGROUND_SYNC_TASKS) == 1
    task = next(iter(sync_module._BACKGROUND_SYNC_TASKS))

    # The anchor must survive a garbage collection while the task runs.
    gc.collect()
    assert task in sync_module._BACKGROUND_SYNC_TASKS

    # Let the task finish; the done-callback must remove it from the anchor set.
    release.set()
    await task
    await asyncio.sleep(0)  # allow the done-callback to run
    assert task not in sync_module._BACKGROUND_SYNC_TASKS
    assert len(sync_module._BACKGROUND_SYNC_TASKS) == 0
