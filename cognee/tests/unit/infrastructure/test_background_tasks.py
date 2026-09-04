"""Process-wide background task registry and its shutdown drain (plan item B6)."""

import asyncio
import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure import background_tasks


@pytest.fixture(autouse=True)
def _clean_registry():
    background_tasks._BACKGROUND_TASKS.clear()
    yield
    background_tasks._BACKGROUND_TASKS.clear()


@pytest.mark.asyncio
async def test_drain_awaits_registered_tasks():
    release = asyncio.Event()
    finished = []

    async def work(name):
        await release.wait()
        finished.append(name)

    for name in ("a", "b"):
        background_tasks.register_background_task(asyncio.create_task(work(name)))
    assert background_tasks.pending_background_tasks() == 2

    async def release_soon():
        await asyncio.sleep(0.01)
        release.set()

    asyncio.create_task(release_soon())
    assert await background_tasks.wait_for_background_tasks(timeout=5) is True
    assert sorted(finished) == ["a", "b"]
    assert background_tasks.pending_background_tasks() == 0
    assert background_tasks._BACKGROUND_TASKS == set()


@pytest.mark.asyncio
async def test_drain_waits_for_tasks_spawned_while_draining():
    order = []

    async def child():
        await asyncio.sleep(0.01)
        order.append("child")

    async def parent():
        await asyncio.sleep(0.01)
        order.append("parent")
        background_tasks.register_background_task(asyncio.create_task(child()))

    background_tasks.register_background_task(asyncio.create_task(parent()))

    assert await background_tasks.wait_for_background_tasks(timeout=5) is True
    assert order == ["parent", "child"]


@pytest.mark.asyncio
async def test_drain_timeout_reports_false_and_cancels_nothing():
    release = asyncio.Event()

    async def slow():
        await release.wait()

    task = background_tasks.register_background_task(asyncio.create_task(slow()))

    assert await background_tasks.wait_for_background_tasks(timeout=0.05) is False
    assert not task.done()
    assert not task.cancelled()

    release.set()
    await task
    await asyncio.sleep(0)
    assert background_tasks.pending_background_tasks() == 0


@pytest.mark.asyncio
async def test_failed_task_does_not_raise_from_drain():
    async def boom():
        raise RuntimeError("background failure")

    background_tasks.register_background_task(asyncio.create_task(boom()))
    assert await background_tasks.wait_for_background_tasks(timeout=5) is True


@pytest.mark.asyncio
async def test_drain_with_nothing_registered_returns_immediately():
    assert await background_tasks.wait_for_background_tasks(timeout=0) is True


@pytest.mark.asyncio
async def test_remember_registers_its_background_tasks(monkeypatch):
    """remember() anchors its tasks in the shared registry, so the drain sees them."""
    remember_module = importlib.import_module("cognee.api.v1.remember.remember")
    release = asyncio.Event()

    async def _noop_setup():
        return None

    async def fake_add(*args, **kwargs):
        return None

    async def fake_cognify(*args, **kwargs):
        await release.wait()
        return {}

    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", _noop_setup)
    monkeypatch.setattr("cognee.api.v1.add.add", fake_add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", fake_cognify)

    result = await remember_module.remember(
        "note",
        dataset_id=uuid4(),
        run_in_background=True,
        self_improvement=False,
        user=SimpleNamespace(id=uuid4()),
    )
    assert result._task in background_tasks._BACKGROUND_TASKS
    assert background_tasks.pending_background_tasks() == 1

    release.set()
    assert await background_tasks.wait_for_background_tasks(timeout=5) is True
    assert result.status == "completed"


def test_public_export():
    import cognee

    assert cognee.wait_for_background_tasks is background_tasks.wait_for_background_tasks


@pytest.mark.asyncio
async def test_api_lifespan_drains_background_tasks_on_shutdown(monkeypatch):
    """The FastAPI shutdown hook waits for registered work before tearing engines down."""
    client_module = importlib.import_module("cognee.api.client")

    async def _noop(*args, **kwargs):
        return None

    # Packages re-export these names, shadowing the submodules — resolve the
    # submodules explicitly (lifespan imports from them at call time).
    for module_name, attr in (
        ("cognee.run_migrations", "run_migrations"),
        ("cognee.modules.users.methods", "get_default_user"),
        ("cognee.modules.cognify.recovery", "recover_stale_cognify_runs_on_startup"),
        ("cognee.shared.utils", "close_telemetry_session"),
    ):
        monkeypatch.setattr(importlib.import_module(module_name), attr, _noop)

    drained = {"called": False, "timeout": None}
    real_wait = background_tasks.wait_for_background_tasks

    async def spy_wait(timeout=None):
        drained["called"] = True
        drained["timeout"] = timeout
        return await real_wait(timeout=timeout)

    monkeypatch.setattr(background_tasks, "wait_for_background_tasks", spy_wait)

    finished = []

    async def work():
        await asyncio.sleep(0.01)
        finished.append("done")

    async with client_module.lifespan(client_module.app):
        background_tasks.register_background_task(asyncio.create_task(work()))

    assert drained["called"] is True
    assert drained["timeout"] == client_module.BACKGROUND_DRAIN_TIMEOUT_SECONDS
    assert finished == ["done"]
