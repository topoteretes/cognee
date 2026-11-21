"""Workers as processes on this machine.

:class:`ProcessWorkerBackend` starts one Python process per shard, at most
``max_workers`` at a time for one run, so a run uses as many cores as it has
workers. The limit is each run's own: runs in progress at the same time start
that many workers each.

A worker is configured by its environment alone, which it inherits from this
process: what this process changed in code (``cognee.config``) does not reach
it. Workers and this process read and write the same databases and file
storage at the same time, which is supported with PostgreSQL as the relational,
vector and graph store only (``ShardedExecutor`` refuses any other run).

A worker is started with the id of its shard and reads everything else from the
shard's rows, where it also stores its answer. Its standard input is a pipe
this process holds open and never writes to: when this process ends, however it
ends, the pipe closes, and the worker cancels its items and exits (see
``worker``). So a worker outlives the process that owns its run only for as
long as that cancellation takes. Nothing makes a process that recovers the run
wait for it: the run's ownership lock, which recovery goes by, was this
process's alone.

A run that is cancelled or fails asks its workers to stop and waits for them,
so that none still writes when the run is rolled back. A worker that has not
ended ``STOP_WAIT_SECONDS`` after it was asked is killed. A worker that is
running its shard is never timed: it is waited for however long it takes.
"""

import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from uuid import UUID

from cognee.modules.pipelines.operations.run_tasks import run_all_or_cancel
from cognee.shared.logging_utils import get_logger

logger = get_logger("process_worker_backend")

WORKER_COMMAND = (sys.executable, "-m", "cognee.modules.pipelines.execution.worker")

# How long a worker that was asked to stop has to end before it is killed.
STOP_WAIT_SECONDS = 60


class ProcessWorkerBackend:
    """``WorkerBackend`` that runs each shard in a process on this machine.

    ``command`` is what starts a worker; the shard's id is appended to it. An
    application whose workers need setup of its own first (a database adapter
    to register) passes a command that does it and then runs ``worker.main``.

    The process ``command`` starts has to be the worker itself, with the
    standard input it is given: it is the process that is asked to stop, waited
    for and killed if it does not end, and its standard input is how it learns
    that this process is gone. A shell or wrapper that runs the worker as a
    child of its own is stopped in its place and leaves it running.
    """

    def __init__(self, max_workers: int, command: Sequence[str] = WORKER_COMMAND):
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        self.max_workers = max_workers
        self.command = list(command)

    async def run_shards(
        self,
        pipeline_run_id: UUID,
        shard_ids: list[UUID],
        on_shard_ended: Callable[[UUID], Awaitable[None]],
    ) -> None:
        free_workers = asyncio.Semaphore(self.max_workers)

        async def run_shard(shard_id: UUID) -> None:
            async with free_workers:
                await self._run_worker(pipeline_run_id, shard_id)
            await on_shard_ended(shard_id)

        # A failure or a cancellation stops every worker before this returns:
        # _run_worker ends its process on the way out.
        await run_all_or_cancel(
            [asyncio.create_task(run_shard(shard_id)) for shard_id in shard_ids]
        )

    async def _run_worker(self, pipeline_run_id: UUID, shard_id: UUID) -> None:
        """One worker for one shard, to its end."""
        worker = await asyncio.create_subprocess_exec(
            *self.command, str(shard_id), stdin=asyncio.subprocess.PIPE
        )
        try:
            exit_code = await worker.wait()
        finally:
            if worker.returncode is None:
                await _to_its_end(_stop(worker, pipeline_run_id, shard_id))

        if exit_code != 0:
            # Whether it stored a result is on the shard's row. Why it failed is
            # in the worker's own log, which goes where this process's does.
            logger.error(
                "Run %s shard %s: the worker exited with code %d.",
                pipeline_run_id,
                shard_id,
                exit_code,
            )


async def _stop(worker: asyncio.subprocess.Process, pipeline_run_id: UUID, shard_id: UUID) -> None:
    """Ask ``worker`` to stop and wait for its end, killing it when it has not
    ended after ``STOP_WAIT_SECONDS``.

    A worker stops by cancelling its items in its event loop, so one whose loop
    is blocked does not hear the request. Killed, it is a worker that died: the
    run it worked for is ending and rolls back what it left unfinished.
    """
    worker.terminate()
    try:
        await asyncio.wait_for(worker.wait(), STOP_WAIT_SECONDS)
    except asyncio.TimeoutError:
        # It may have ended just as the wait ran out, and there is then no
        # process to kill.
        if worker.returncode is None:
            logger.error(
                "Run %s shard %s: the worker did not end within %d seconds of being asked"
                " to stop and is killed.",
                pipeline_run_id,
                shard_id,
                STOP_WAIT_SECONDS,
            )
            worker.kill()
        await worker.wait()


async def _to_its_end(work: Awaitable[object]) -> None:
    """Await ``work`` to its end, however often this task is cancelled meanwhile.

    ``asyncio.shield`` alone lets go at the second cancellation (a shutdown
    cancels more than once). The cancellation is passed on once ``work`` is done.
    """
    task = asyncio.ensure_future(work)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError
