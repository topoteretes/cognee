"""Guards the strong-reference anchor for background remember tasks (#4312).

``remember`` launches its background work (permanent add+cognify run, or the
session→graph bridge) with ``asyncio.create_task`` and keeps the task only on
``RememberResult._task``. The API router serializes the result and drops it,
and the event loop keeps only a *weak* reference to a task — so unless the
task is anchored elsewhere it can be garbage-collected mid-flight, silently
aborting the run while the HTTP call already returned success. The fix
anchors it in the module-level ``_BACKGROUND_REMEMBER_TASKS`` set and
discards it on completion (same pattern as #3251).
"""

import asyncio
import gc
import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

remember_module = importlib.import_module("cognee.api.v1.remember.remember")


@pytest.fixture(autouse=True)
def _no_db_setup(monkeypatch):
    """remember() initializes the database before doing anything else."""

    async def _noop_setup():
        return None

    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", _noop_setup)


@pytest.mark.asyncio
async def test_background_remember_task_is_anchored_until_done(monkeypatch):
    release = asyncio.Event()

    async def fake_add(*args, **kwargs):
        return None

    async def fake_cognify(*args, **kwargs):
        await release.wait()
        return {}

    monkeypatch.setattr("cognee.api.v1.add.add", fake_add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", fake_cognify)

    result = await remember_module.remember(
        "note",
        dataset_id=uuid4(),
        run_in_background=True,
        self_improvement=False,
        user=SimpleNamespace(id=uuid4()),
    )

    assert result.status == "running"
    task = result._task
    assert task in remember_module._BACKGROUND_REMEMBER_TASKS

    # The router drops the RememberResult after serializing it; the anchor must
    # keep the task alive through a garbage-collection pass.
    del result
    gc.collect()
    await asyncio.sleep(0)
    assert not task.done()
    assert task in remember_module._BACKGROUND_REMEMBER_TASKS

    # Let the task finish; the done-callback must remove it from the anchor set.
    release.set()
    await task
    await asyncio.sleep(0)
    assert task not in remember_module._BACKGROUND_REMEMBER_TASKS


@pytest.mark.asyncio
async def test_session_bridge_task_is_anchored_until_done(monkeypatch):
    release = asyncio.Event()

    async def fake_add_to_session(session_id, data, user):
        return None

    async def fake_improve(*args, **kwargs):
        await release.wait()

    # The improve package re-exports the improve() function, shadowing the
    # submodule — patch the package attribute directly.
    improve_pkg = importlib.import_module("cognee.api.v1.improve")
    monkeypatch.setattr(remember_module, "_add_to_session", fake_add_to_session)
    monkeypatch.setattr(improve_pkg, "improve", fake_improve)

    result = await remember_module.remember(
        "note",
        dataset_id=uuid4(),
        session_id="session-4312",
        self_improvement=True,
        user=SimpleNamespace(id=uuid4()),
    )

    assert result.status == "session_stored"
    task = result._task
    assert task in remember_module._BACKGROUND_REMEMBER_TASKS

    del result
    gc.collect()
    await asyncio.sleep(0)
    assert not task.done()
    assert task in remember_module._BACKGROUND_REMEMBER_TASKS

    release.set()
    await task
    await asyncio.sleep(0)
    assert task not in remember_module._BACKGROUND_REMEMBER_TASKS
