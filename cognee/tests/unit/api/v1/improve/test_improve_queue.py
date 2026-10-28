"""Improves that share a session or dataset queue one after another (#5566),
like pipeline runs on the per-dataset lock. Nothing is skipped for contention."""

import asyncio
import types
from uuid import uuid4

import pytest

from cognee.modules.improve.result import StageResult

from .conftest import FakeStage


async def _start_holder(harness, calls, *, session_ids=None):
    """A background improve blocked inside its first stage until ``gate`` is set."""
    gate = asyncio.Event()

    async def slow_stage(_inputs):
        await gate.wait()
        return StageResult.completed("slow", items=1)

    harness.use_stages(
        [FakeStage("slow", run=slow_stage, calls=calls), FakeStage("after", calls=calls)]
    )
    holder = await harness.improve(session_ids=session_ids, run_in_background=True)
    await asyncio.sleep(0)  # let the task reach the blocked stage
    assert holder.status == "running"
    return holder, gate


async def _settle():
    for _ in range(10):
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("holder_sessions", "waiter_sessions"),
    [
        (["chat_1"], ["chat_2"]),  # #5566: another session on the same dataset
        (["chat_1"], ["chat_1"]),  # the same session
        (None, None),  # dataset-only
    ],
    ids=["different-session", "same-session", "dataset-only"],
)
async def test_an_overlapping_improve_waits_then_runs_every_stage(
    harness, holder_sessions, waiter_sessions
):
    calls = []
    holder, gate = await _start_holder(harness, calls, session_ids=holder_sessions)

    waiter = asyncio.create_task(harness.improve(session_ids=waiter_sessions))
    await _settle()
    assert not waiter.done()
    assert calls == ["slow"]  # the waiter has not started

    gate.set()
    await holder.wait()
    result = await asyncio.wait_for(waiter, 1)

    assert calls == ["slow", "after", "slow", "after"]
    assert result.status == "completed"
    assert [stage.stage for stage in result.stages] == ["slow", "after"]


@pytest.mark.asyncio
async def test_a_background_improve_returns_at_once_and_runs_after_the_holder(harness):
    calls = []
    holder, gate = await _start_holder(harness, calls, session_ids=["chat_1"])

    queued = await harness.improve(session_ids=["chat_2"], run_in_background=True)
    assert queued.status == "running"
    await _settle()
    assert calls == ["slow"]

    gate.set()
    await holder.wait()
    await asyncio.wait_for(queued.wait(), 1)
    assert calls == ["slow", "after", "slow", "after"]
    assert queued.status == "completed"


@pytest.mark.asyncio
async def test_improves_on_different_datasets_do_not_wait(harness):
    calls = []
    holder, gate = await _start_holder(harness, calls, session_ids=["chat_1"])
    # The harness resolves whatever dataset it holds at call time.
    harness.dataset = types.SimpleNamespace(id=uuid4(), name="other", owner_id=harness.user.id)

    harness.use_stages([FakeStage("a", calls=calls)])
    result = await asyncio.wait_for(harness.improve(dataset="other"), 1)

    assert result.status == "completed"
    gate.set()
    await holder.wait()


@pytest.mark.asyncio
async def test_a_cancelled_holder_lets_the_next_improve_run(harness):
    calls = []
    holder, _gate = await _start_holder(harness, calls, session_ids=["chat_1"])
    waiter = asyncio.create_task(harness.improve(session_ids=["chat_2"]))
    await _settle()

    # Swapped before the cancel: the waiter picks its stages up as soon as it
    # is let in, and the holder's blocked stage would never finish.
    harness.use_stages([FakeStage("a", calls=calls)])
    holder._task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder._task

    result = await asyncio.wait_for(waiter, 1)
    assert result.status == "completed"
