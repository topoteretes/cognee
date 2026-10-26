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


class ProcessOutputError(RuntimeError):
    """A process emitted output that requires terminating its transfer."""


async def run_process(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: float,
    reject_stderr: bytes | None = None,
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
    rejection = asyncio.Event()
    rejected = None

    async def read_stderr(stream, needle: bytes):
        chunks = []
        tail = b""
        while chunk := await stream.read(65536):
            chunks.append(chunk)
            combined = tail + chunk
            if needle in combined:
                rejection.set()
            tail = combined[-len(needle) :]
        return b"".join(chunks)

    async def communicate(process):
        stdout, stderr = await asyncio.gather(
            process.stdout.read(), read_stderr(process.stderr, reject_stderr or b"")
        )
        await process.wait()
        return stdout, stderr

    try:
        process = await asyncio.shield(spawning)
        if reject_stderr is None:
            communication = asyncio.create_task(process.communicate())
        else:
            communication = asyncio.create_task(communicate(process))
            rejected = asyncio.create_task(rejection.wait())
            done, _ = await asyncio.wait(
                {communication, rejected}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if rejection.is_set():
                raise ProcessOutputError("The Git server ignored the partial clone filter.")
            if not done:
                raise asyncio.TimeoutError
        stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout)
    except (asyncio.CancelledError, asyncio.TimeoutError, ProcessOutputError):
        await finish_cleanup(stop_process_tree(spawning, communication))
        raise
    finally:
        if rejected is not None:
            rejected.cancel()
            await finish_cleanup(asyncio.gather(rejected, return_exceptions=True))
    return await process.wait(), stdout, stderr
