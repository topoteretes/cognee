"""``ProcessWorkerBackend`` runs each shard in a process of its own on this machine.

The workers here are real processes running a small stand-in script, so what is
tested is the backend's own work: that each shard gets a worker started with
its id, how many run at a time, and that none is left running. What a worker
does with its shard is ``run_pipeline_shard`` and is tested there.
"""

import asyncio
import json
import os
import sys
import textwrap
import time
from uuid import uuid4

import pytest

from cognee.modules.pipelines.execution import process_backend
from cognee.modules.pipelines.execution.process_backend import WORKER_COMMAND, ProcessWorkerBackend

# Stand-in worker: ``python worker.py <notes folder> <shard id>``. What it does
# is chosen by its shard id, looked up in the notes folder's ``modes.json``
# (default: note the id it was started with and exit 0). Asked to stop, "hang"
# ends at once, "slow_stop" takes half a second and notes that it stopped, and
# "deaf" does not hear it.
WORKER = textwrap.dedent(
    """
    import json, os, signal, sys, time

    notes, shard_id = sys.argv[1:3]
    mode = json.load(open(os.path.join(notes, "modes.json"))).get(shard_id, "ok")

    asked_to_stop = []
    if mode == "deaf":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "slow_stop":
        signal.signal(signal.SIGTERM, lambda *_: asked_to_stop.append(True))

    running = os.path.join(notes, f"running-{os.getpid()}")
    open(running, "w").write(shard_id)
    at_start = len([name for name in os.listdir(notes) if name.startswith("running-")])
    open(os.path.join(notes, f"seen-{os.getpid()}"), "w").write(str(at_start))

    if mode == "hang":
        while True:
            time.sleep(0.05)
    if mode == "deaf":
        # It ends by itself, so a backend that only waits for it fails its test.
        time.sleep(30)
        sys.exit(0)
    if mode == "slow_stop":
        while not asked_to_stop:
            time.sleep(0.05)
        time.sleep(0.5)
        open(os.path.join(notes, f"stopped-{os.getpid()}"), "w").close()
        sys.exit(0)
    time.sleep(0.3)
    os.remove(running)
    if mode == "crash":
        sys.exit(3)
    open(os.path.join(notes, f"done-{shard_id}"), "w").close()
    """
)


@pytest.fixture
def workers(tmp_path):
    """A backend factory whose workers run the stand-in script, and their notes folder."""
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    notes = tmp_path / "notes"
    notes.mkdir()

    def backend(max_workers, modes=None):
        (notes / "modes.json").write_text(json.dumps(modes or {}))
        return ProcessWorkerBackend(max_workers, [sys.executable, str(script), str(notes)])

    backend.notes = notes
    return backend


def _shard_ids(count):
    return [uuid4() for _ in range(count)]


async def _run(backend, shard_ids):
    ended = []

    async def on_shard_ended(shard_id):
        ended.append(shard_id)

    await backend.run_shards(uuid4(), shard_ids, on_shard_ended)
    return ended


def _done(notes) -> set[str]:
    return {path.name.removeprefix("done-") for path in notes.glob("done-*")}


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_the_default_worker_is_this_interpreter_running_the_worker_module():
    assert WORKER_COMMAND == (sys.executable, "-m", "cognee.modules.pipelines.execution.worker")
    assert ProcessWorkerBackend(4).command == list(WORKER_COMMAND)


def test_a_backend_runs_at_least_one_worker():
    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        ProcessWorkerBackend(0)


@pytest.mark.asyncio
async def test_each_shard_gets_a_worker_started_with_its_id_and_nothing_else(workers):
    """A backend carries no work and no result: the worker reads its shard from
    its rows, by the id it is started with."""
    shard_ids = _shard_ids(3)

    ended = await _run(workers(3), shard_ids)

    assert _done(workers.notes) == {str(shard_id) for shard_id in shard_ids}
    assert sorted(ended, key=str) == sorted(shard_ids, key=str)


@pytest.mark.asyncio
async def test_a_worker_that_fails_ends_its_own_shard_only(workers):
    """Whether a worker stored a result is on its shard's row, which the run
    reads afterwards. To the backend a failed worker is one that has ended."""
    shard_ids = _shard_ids(3)

    ended = await _run(workers(3, {str(shard_ids[1]): "crash"}), shard_ids)

    assert _done(workers.notes) == {str(shard_ids[0]), str(shard_ids[2])}
    assert sorted(ended, key=str) == sorted(shard_ids, key=str)


@pytest.mark.asyncio
async def test_no_more_than_max_workers_run_at_a_time(workers):
    shard_ids = _shard_ids(6)

    await _run(workers(2), shard_ids)

    assert len(_done(workers.notes)) == 6
    seen = [int(path.read_text()) for path in workers.notes.glob("seen-*")]
    assert len(seen) == 6
    # Two ran together, and never a third.
    assert max(seen) == 2


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="reads process state with POSIX signals")
async def test_a_cancelled_run_leaves_no_worker_running(workers):
    shard_ids = _shard_ids(3)
    backend = workers(3, {str(shard_id): "hang" for shard_id in shard_ids})

    run = asyncio.create_task(_run(backend, shard_ids))
    pids = await _worker_pids(workers.notes, 3)
    assert all(_alive(pid) for pid in pids)

    # A shutdown cancels more than once.
    run.cancel()
    await asyncio.sleep(0)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert not any(_alive(pid) for pid in pids)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="reads process state with POSIX signals")
async def test_a_failing_run_leaves_no_worker_running(workers):
    shard_ids = _shard_ids(2)
    backend = workers(2, {str(shard_ids[0]): "hang"})

    async def on_shard_ended(shard_id):
        # The second shard's worker has ended; the first one's hangs.
        raise RuntimeError("could not record progress")

    run = asyncio.create_task(backend.run_shards(uuid4(), shard_ids, on_shard_ended))
    pids = await _worker_pids(workers.notes, 2)

    with pytest.raises(RuntimeError, match="could not record progress"):
        await run

    assert not any(_alive(pid) for pid in pids)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="reads process state with POSIX signals")
async def test_a_worker_that_does_not_stop_when_asked_is_killed(workers, monkeypatch):
    """A worker hears the request to stop in its event loop, which can be
    blocked. A run that is ending does not wait for such a worker for good."""
    monkeypatch.setattr(process_backend, "STOP_WAIT_SECONDS", 1)
    shard_ids = _shard_ids(2)
    backend = workers(2, {str(shard_ids[0]): "deaf", str(shard_ids[1]): "hang"})

    run = asyncio.create_task(_run(backend, shard_ids))
    pids = await _worker_pids(workers.notes, 2)

    cancelled_at = time.monotonic()
    run.cancel()
    # A shutdown cancels more than once, here also while the worker is waited for.
    await asyncio.sleep(0.2)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert not any(_alive(pid) for pid in pids)
    # Left alone, the deaf worker ends after 30 seconds.
    assert time.monotonic() - cancelled_at < 15


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="reads process state with POSIX signals")
async def test_a_worker_that_takes_time_to_stop_is_waited_for(workers):
    """The wait is a worker's time to end the cancellation of its items: one
    that is still stopping is not killed."""
    shard_ids = _shard_ids(1)
    backend = workers(1, {str(shard_ids[0]): "slow_stop"})

    run = asyncio.create_task(_run(backend, shard_ids))
    (pid,) = await _worker_pids(workers.notes, 1)

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert (workers.notes / f"stopped-{pid}").exists()
    assert not _alive(pid)


async def _worker_pids(notes, count) -> list[int]:
    """The pids of ``count`` workers, once that many have started."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        pids = [int(path.name.removeprefix("seen-")) for path in notes.glob("seen-*")]
        if len(pids) == count:
            return pids
        await asyncio.sleep(0.02)
    raise AssertionError(f"{count} workers did not start")
