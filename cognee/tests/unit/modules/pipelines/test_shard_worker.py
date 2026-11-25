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

# The worker's own ``main``, with the shard replaced by one that never ends.
NEVER_ENDING_WORKER = textwrap.dedent(
    """
    import asyncio, sys
    from uuid import uuid4
    import cognee.modules.pipelines.execution.worker as worker

    async def never_ends(shard_id):
        print("running", flush=True)
        await asyncio.sleep(3600)

    worker.run_pipeline_shard = never_ends
    try:
        asyncio.run(worker.main(uuid4()))
    except asyncio.CancelledError:
        sys.exit(7)
    """
)


@pytest.mark.asyncio
async def test_a_worker_runs_the_shard_it_was_started_with(monkeypatch):
    run_pipeline_shard = AsyncMock()
    monkeypatch.setattr(worker, "run_pipeline_shard", run_pipeline_shard)
    # Standard input is not a pipe an owner holds here, so the watch for it is
    # left out: it would stop the worker at once.
    monkeypatch.setattr(worker, "cancel_when_owner_is_gone", lambda task, loop: None)
    shard_id = uuid4()

    await worker.main(shard_id)

    run_pipeline_shard.assert_awaited_once_with(shard_id)


def _start_never_ending_worker():
    process = subprocess.Popen(
        [sys.executable, "-c", NEVER_ENDING_WORKER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout.readline().strip() == "running"
    return process


def test_a_worker_stops_when_its_owner_is_gone():
    process = _start_never_ending_worker()
    try:
        # What the operating system does to the pipe when the owner's process ends.
        process.stdin.close()

        assert process.wait(timeout=60) == 7
    finally:
        process.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows ends a terminated process outright")
def test_a_worker_asked_to_stop_cancels_its_shard():
    process = _start_never_ending_worker()
    try:
        process.send_signal(signal.SIGTERM)

        assert process.wait(timeout=60) == 7
    finally:
        process.kill()
