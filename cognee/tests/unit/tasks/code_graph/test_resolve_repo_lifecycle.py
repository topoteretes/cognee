"""Cancellation must finish Git and release the clone before another caller can use it."""

import asyncio
import importlib
import os
import signal
import sys
from pathlib import Path

import pytest
from filelock import FileLock

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")


async def _wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=5)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Container PID 1 may leave an already dead orphan as a zombie for a while.
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return not sys.platform.startswith("linux")


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="Checks POSIX process-group termination")
@pytest.mark.parametrize(
    "interruption", ["cancel", "timeout", "cancel-during-spawn", "parent-exited"]
)
async def test_interrupted_git_reaps_parent_and_stops_descendants(
    tmp_path, monkeypatch, interruption
):
    ready = tmp_path / "pids"
    script = (
        "import os, subprocess, sys, time; from pathlib import Path; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "Path(sys.argv[1]).write_text(str(os.getpid()) + ' ' + str(child.pid)); "
        "sys.stdout.write('x' * 262144); sys.stdout.flush(); "
        + ("" if interruption == "parent-exited" else "time.sleep(120)")
    )
    # A Git stand-in with a real helper inheriting its stdout/stderr pipes.
    monkeypatch.setattr(resolve_module.shutil, "which", lambda name: sys.executable)
    monkeypatch.setattr(resolve_module, "_GIT_TIMEOUT_SECONDS", 1)
    real_spawn = asyncio.create_subprocess_exec
    processes = []
    allow_spawn_return = asyncio.Event()

    async def spawn(*args, **kwargs):
        process = await real_spawn(*args, **kwargs)
        processes.append(process)
        if interruption == "cancel-during-spawn":
            await allow_spawn_return.wait()
        return process

    monkeypatch.setattr(resolve_module.asyncio, "create_subprocess_exec", spawn)
    running = asyncio.create_task(resolve_module._exec_git(["-c", script, str(ready)]))
    pids = []
    try:
        await _wait_until(lambda: ready.exists() and len(ready.read_text().split()) == 2)
        pids = [int(pid) for pid in ready.read_text().split()]
        if interruption in ("cancel", "cancel-during-spawn"):
            running.cancel()
            await asyncio.sleep(0)
            allow_spawn_return.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, timeout=5)
        else:
            with pytest.raises(resolve_module.CodeRepositoryError, match="timed out"):
                await asyncio.wait_for(running, timeout=5)

        assert processes[0].returncode is not None, "Git must be reaped before returning"
        await _wait_until(lambda: all(not _alive(pid) for pid in pids))
    finally:
        allow_spawn_return.set()
        for pid in pids:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        for process in processes:
            await process.communicate()


@pytest.mark.asyncio
async def test_lock_wait_times_out_without_entering(tmp_path, monkeypatch):
    target = tmp_path / "repo"
    lock_dir = tmp_path / resolve_module._LOCKS_DIR
    lock_dir.mkdir()
    monkeypatch.setattr(resolve_module, "_GIT_TIMEOUT_SECONDS", 0.1)

    async def acquire():
        async with resolve_module._clone_lock(target):
            pytest.fail("Entered an already held clone lock")

    with (
        FileLock(str(lock_dir / "repo.lock")),
        pytest.raises(resolve_module.CodeRepositoryError, match="lock"),
    ):
        await asyncio.wait_for(acquire(), timeout=1)

    async with resolve_module._clone_lock(target):
        assert (lock_dir / "repo.lock").exists()


@pytest.mark.asyncio
async def test_cancelled_lock_waiter_does_not_take_the_lock_later(tmp_path):
    target = tmp_path / "repo"
    waiting = asyncio.Event()

    async def waiter():
        waiting.set()
        async with resolve_module._clone_lock(target):
            pytest.fail("Cancelled waiter acquired the lock")

    async with resolve_module._clone_lock(target):
        task = asyncio.create_task(waiter())
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async with resolve_module._clone_lock(target):
        pass
