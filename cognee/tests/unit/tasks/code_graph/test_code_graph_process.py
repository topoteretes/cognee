"""Actual subprocess cancellation, including the spawn and cleanup windows."""

import asyncio
import importlib
import os
import sys

import pytest

runner = importlib.import_module("cognee.tasks.code_graph.process")


@pytest.mark.asyncio
async def test_run_process_returns_output():
    result = await runner.run_process([sys.executable, "-c", "print('ok')"], timeout=5)
    assert result == (0, b"ok\n", b"")


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [False, True])
async def test_cancellation_during_spawn_reaps_process(monkeypatch, output):
    original = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    release = asyncio.Event()
    processes = []

    async def delayed_spawn(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        spawned.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    code = "import time, sys; "
    if output:
        code += "sys.stdout.write('x' * 1000000); sys.stdout.flush(); "
    code += "time.sleep(60)"
    task = asyncio.create_task(runner.run_process([sys.executable, "-c", code], timeout=5))
    await asyncio.wait_for(spawned.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert processes[0].returncode is not None


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="POSIX process group inspection")
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancellation_stop_child_processes(tmp_path, cancel):
    pid_file = tmp_path / "pids"
    code = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "open(sys.argv[1], 'w').write(str(os.getpid())+' '+str(child.pid))\n"
        "time.sleep(60)\n"
    )
    task = asyncio.create_task(
        runner.run_process(
            [sys.executable, "-c", code, str(pid_file)],
            timeout=0.5 if not cancel else 30,
        )
    )
    try:

        async def wait_for_pid():
            while not pid_file.exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_pid(), 5)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else asyncio.TimeoutError):
            await asyncio.wait_for(task, 5)
        pids = [int(pid) for pid in pid_file.read_text().split()]
        for pid in pids:
            # Zombies cannot execute or write files and are waiting for init to reap.
            process = await asyncio.create_subprocess_exec(
                "ps",
                "-o",
                "stat=",
                "-p",
                str(pid),
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            assert not stdout.strip() or stdout.strip().startswith(b"Z")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
