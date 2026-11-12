"""The worker process of ``ProcessWorkerBackend`` runs one shard and stops with its owner.

A run is rolled back when it fails, or when its owner is gone and another
process recovers it, so a worker must never keep writing after it was asked to
stop or its owner went away.
"""

import signal
import subprocess
import sys
import textwrap
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.pipelines.execution import worker

# The worker's own ``main`` with the shard replaced, and how long it has to end
# once its owner is gone.
WORKER = textwrap.dedent(
    """
    import asyncio, sys, time
    from uuid import uuid4
    import cognee.modules.pipelines.execution.worker as worker

    worker.OWNER_GONE_WAIT_SECONDS = {wait}

    {shard}

    worker.run_pipeline_shard = shard
    try:
        asyncio.run(worker.main(uuid4()))
    except asyncio.CancelledError:
        sys.exit(7)
    """
)

NEVER_ENDS = """
async def shard(shard_id):
    print("running", flush=True)
    await asyncio.sleep(3600)
"""

# Synchronous code in the event loop: nothing the loop is asked reaches the shard.
BLOCKS_ITS_EVENT_LOOP = """
async def shard(shard_id):
    print("running", flush=True)
    time.sleep(3600)
"""

BLOCKS_ITS_EVENT_LOOP_THEN_ENDS = """
async def shard(shard_id):
    print("running", flush=True)
    time.sleep(4)
    print("ended", flush=True)
"""

TAKES_TIME_TO_CANCEL = """
async def shard(shard_id):
    print("running", flush=True)
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await asyncio.sleep(2)
        print("cancelled", flush=True)
        raise
"""


@pytest.mark.asyncio
async def test_a_worker_runs_the_shard_it_was_started_with(monkeypatch):
    run_pipeline_shard = AsyncMock()
    monkeypatch.setattr(worker, "run_pipeline_shard", run_pipeline_shard)
    # Standard input is not a pipe an owner holds here, so the watch for it is
    # left out: it would stop the worker at once.
    monkeypatch.setattr(worker, "stop_when_owner_is_gone", lambda task, loop: None)
    shard_id = uuid4()

    await worker.main(shard_id)

    run_pipeline_shard.assert_awaited_once_with(shard_id)


def _start_worker(shard: str = NEVER_ENDS, wait: float = 60):
    process = subprocess.Popen(
        [sys.executable, "-c", WORKER.format(shard=shard, wait=wait)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout.readline().strip() == "running"
    return process


def test_a_worker_stops_when_its_owner_is_gone():
    process = _start_worker()
    try:
        # What the operating system does to the pipe when the owner's process ends.
        process.stdin.close()

        assert process.wait(timeout=60) == 7
    finally:
        process.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows ends a terminated process outright")
def test_a_worker_asked_to_stop_cancels_its_shard():
    process = _start_worker()
    try:
        process.send_signal(signal.SIGTERM)

        assert process.wait(timeout=60) == 7
    finally:
        process.kill()


def test_a_worker_that_does_not_hear_that_its_owner_is_gone_ends_after_the_wait():
    """The cancellation is heard in the event loop. A worker whose loop is
    blocked must not outlive its owner for good: the run is rolled back by
    whichever process recovers it."""
    process = _start_worker(BLOCKS_ITS_EVENT_LOOP, wait=1)
    try:
        process.stdin.close()

        assert process.wait(timeout=30) == 1
    finally:
        process.kill()


def test_a_worker_whose_owner_is_there_is_never_timed():
    """The wait starts when the owner goes away. A worker whose event loop is
    blocked for longer than the wait, with its owner there, runs to its end."""
    process = _start_worker(BLOCKS_ITS_EVENT_LOOP_THEN_ENDS, wait=1)
    try:
        assert process.wait(timeout=60) == 0
        assert process.stdout.read().strip() == "ended"
    finally:
        process.kill()


def test_a_worker_that_is_ending_its_cancellation_is_left_to_end_it():
    """The wait is a worker's time to end the cancellation of its items."""
    process = _start_worker(TAKES_TIME_TO_CANCEL, wait=30)
    try:
        process.stdin.close()

        assert process.wait(timeout=60) == 7
        assert process.stdout.read().strip() == "cancelled"
    finally:
        process.kill()
