"""Worker entry point of ``ProcessWorkerBackend``:
``python -m cognee.modules.pipelines.execution.worker <shard id>``.

Takes the shard, runs its data items and stores how each ended on the shard's
rows (``run_pipeline_shard``). Exit code 0 means the result is stored, a result
in which items failed included. Anything else means the shard could not run or
could not store its result; the run's owner then reads which items completed
from their stamps.

The process that started this one holds the other end of its standard input
and never writes to it. When that end closes, the run's owner is gone and this
process stops: it cancels its items and exits once they have ended, because the
run is rolled back by whichever process recovers it, and nothing should still
be writing for the run then. Started with a standard input nobody holds open (a
closed one, ``/dev/null``), it therefore stops at once.

The cancellation is heard in this process's event loop, so a worker whose loop
is blocked does not hear it. A worker that is still running
``OWNER_GONE_WAIT_SECONDS`` after its owner went away therefore ends outright.
Nothing here is timed while the owner is there: its end of standard input stays
open however busy either process is, and the wait starts only when it closes.
"""

import asyncio
import os
import signal
import sys
import threading
import time
from uuid import UUID

# Importing cognee is what registers its pipelines' portable task resolvers
# (see ``task_resolvers``), which this process rebuilds a shard's tasks from.
import cognee

from .run_pipeline_shard import run_pipeline_shard

# How long a worker whose owner is gone has to end before it ends outright.
OWNER_GONE_WAIT_SECONDS = 60


def stop_when_owner_is_gone(task: asyncio.Task, loop: asyncio.AbstractEventLoop) -> None:
    """Cancel ``task`` once standard input closes, which it does when the process
    holding its other end has ended, and end this process if it is still
    running ``OWNER_GONE_WAIT_SECONDS`` later.

    The watch is a thread, so it also sees the owner go while the event loop is
    blocked, which is when the cancellation is not heard.
    """

    def watch() -> None:
        try:
            # The file descriptor, not sys.stdin: a thread blocked in a buffered
            # read holds a lock the interpreter needs to exit.
            while os.read(0, 1024):
                pass
        except OSError:
            pass
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            pass  # The loop is closed: the shard has already ended.
        # A worker that heard the cancellation has exited by now, and this
        # thread with it. Nothing is written first: a write that blocks would
        # keep the process.
        time.sleep(OWNER_GONE_WAIT_SECONDS)
        os._exit(1)

    threading.Thread(target=watch, name="owner-watch", daemon=True).start()


async def main(shard_id: UUID) -> None:
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    assert task is not None
    if sys.platform != "win32":
        # Asked to stop: leave no item writing (run_pipeline_shard waits for its
        # items to end their cancellation).
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    stop_when_owner_is_gone(task, loop)
    await run_pipeline_shard(shard_id)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m cognee.modules.pipelines.execution.worker <shard id>")
    asyncio.run(main(UUID(sys.argv[1])))
