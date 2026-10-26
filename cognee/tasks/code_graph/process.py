"""Subprocess lifetime shared by Git and the code extractor."""

import asyncio
import os
import signal
from pathlib import Path


async def finish_cleanup(awaitable):
    """Finish owned cleanup even when the caller is cancelled repeatedly."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def stop_process_tree(spawning, communication) -> None:
    """Stop the spawned process group and drain its pipes before returning.

    POSIX groups cover children after their parent exits. Windows taskkill only
    covers a still-live parent; equivalent ownership there needs Job Objects.
    """
    process = await spawning
    if communication is None:
        communication = asyncio.create_task(process.communicate())
    if os.name == "nt":
        if process.returncode is None:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(process.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The process group may have finished while cancellation was delivered.
            pass
    await process.wait()
    await communication


async def run_process(
    command: list[str], *, cwd: Path | None = None, env: dict | None = None, timeout: float
) -> tuple[int, bytes, bytes]:
    """Run a command; timeout and cancellation reap its children before raising."""
    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
    )
    communication = None
    try:
        process = await asyncio.shield(spawning)
        communication = asyncio.create_task(process.communicate())
        stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        await finish_cleanup(stop_process_tree(spawning, communication))
        raise
    return await process.wait(), stdout, stderr
