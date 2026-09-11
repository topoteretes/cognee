"""Recovery of abandoned pipeline runs must not hold the port closed (SDK-577).

The sweep used to be awaited inside the lifespan, so a boot that found work to
do served nothing until it finished. It runs as a background task now, and
these pin the two halves of that: startup does not wait for it, and shutdown
does not walk away from it while the database engines are being torn down.
"""

import asyncio
import importlib
from contextlib import AsyncExitStack

import pytest

# Several of these names are rebound to functions by their parent packages, so
# the modules themselves have to come from importlib for monkeypatch to reach
# the attributes the lifespan reads.
client_module = importlib.import_module("cognee.api.client")
recovery_module = importlib.import_module("cognee.modules.pipelines.recovery")
user_methods = importlib.import_module("cognee.modules.users.methods")
migrations_module = importlib.import_module("cognee.run_migrations")


@pytest.fixture
def quiet_lifespan(monkeypatch):
    """Everything the lifespan does apart from recovery, stubbed out."""

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(migrations_module, "run_migrations", _noop)
    monkeypatch.setattr(user_methods, "get_default_user", _noop)


@pytest.mark.asyncio
async def test_startup_does_not_wait_for_recovery(quiet_lifespan, monkeypatch):
    recovery_started = asyncio.Event()
    let_recovery_finish = asyncio.Event()
    outcome = []

    async def _slow_recovery():
        recovery_started.set()
        await let_recovery_finish.wait()
        outcome.append("finished")

    monkeypatch.setattr(recovery_module, "recover_abandoned_pipeline_runs", _slow_recovery)

    async with AsyncExitStack() as stack:
        # Entered under a timeout on purpose: a regression that awaits the
        # sweep inside the lifespan would otherwise deadlock this test (the
        # sweep waits for an event this body sets) and hang CI until the
        # per-test timeout, instead of failing here in seconds.
        await asyncio.wait_for(
            stack.enter_async_context(client_module.lifespan(client_module.app)), timeout=5
        )

        await asyncio.wait_for(recovery_started.wait(), timeout=5)
        assert outcome == []

        let_recovery_finish.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert outcome == ["finished"]


@pytest.mark.asyncio
async def test_shutdown_cancels_an_unfinished_recovery(quiet_lifespan, monkeypatch):
    """The sweep only closes runs that still have no terminal row, so cancelling
    it costs the work in flight and nothing else. Letting it run into the engine
    teardown below would cost more."""
    recovery_started = asyncio.Event()
    outcome = []

    async def _never_finishing_recovery():
        recovery_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise

    monkeypatch.setattr(
        recovery_module, "recover_abandoned_pipeline_runs", _never_finishing_recovery
    )

    async with client_module.lifespan(client_module.app):
        await asyncio.wait_for(recovery_started.wait(), timeout=5)

    assert outcome == ["cancelled"]


@pytest.mark.asyncio
async def test_a_failed_recovery_is_reported_and_does_not_fail_the_boot(
    quiet_lifespan, monkeypatch, caplog
):
    """A background task's exception is otherwise only surfaced by asyncio's
    "never retrieved" warning at garbage collection. The boot is already over
    by then, so the task boundary reports it instead."""

    async def _failing_recovery():
        raise RuntimeError("relational database is unreachable")

    monkeypatch.setattr(recovery_module, "recover_abandoned_pipeline_runs", _failing_recovery)

    with caplog.at_level("ERROR"):
        async with client_module.lifespan(client_module.app):
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    assert any("relational database is unreachable" in record.message for record in caplog.records)
