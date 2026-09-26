"""Event-loop-agnostic async mutual exclusion.

``asyncio.Lock`` binds to the first event loop that awaits it, so any lock
cached beyond one loop's lifetime — module-level registries, locks on engine
instances held in cross-loop caches — breaks for embedders that run cognee
operations on multiple loops (``asyncio.run`` per operation from worker
threads is a common pattern). The failure is nasty on two counts: an
UNCONTENDED cross-loop acquire silently succeeds (the fast path never
touches the loop), so the bug hides until real concurrency arrives; and a
contended one raises ``RuntimeError: ... is bound to a different event
loop``, wedging the resource until process restart when the holder's loop
is already gone.

:class:`LoopAgnosticLock` mirrors the slice of the ``asyncio.Lock``
interface the codebase uses (``async with``, awaitable ``acquire()``,
``release()``, ``locked()``) but is backed by a ``threading.Lock``, so it
works from any loop or thread AND genuinely serializes across them.
Waiting never blocks the caller's loop: the blocking acquire runs on a
dedicated executor, so parked waiters cannot starve asyncio's shared
default (``to_thread``) executor.
"""

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from cognee.shared.logging_utils import get_logger

logger = get_logger("loop_agnostic_lock")

_lock_wait_executor: ThreadPoolExecutor | None = None
_lock_wait_executor_guard = threading.Lock()


def _get_lock_wait_executor() -> ThreadPoolExecutor:
    global _lock_wait_executor
    with _lock_wait_executor_guard:
        if _lock_wait_executor is None:
            _lock_wait_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="lock-wait")
        return _lock_wait_executor


def _release_abandoned_acquire(lock: threading.Lock):
    """Done-callback releasing a lock acquired by a cancelled waiter."""

    def _release(future: Future) -> None:
        try:
            if future.result():
                lock.release()
        except Exception:
            logger.exception("Failed to release a lock abandoned by cancellation")

    return _release


class LoopAgnosticLock:
    """Async mutual exclusion usable from any event loop or thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    async def acquire(self) -> bool:
        loop = asyncio.get_running_loop()
        if self._lock.acquire(blocking=False):
            # Uncontended fast path: no executor round-trip.
            return True
        future = loop.run_in_executor(_get_lock_wait_executor(), self._lock.acquire)
        try:
            return await future
        except asyncio.CancelledError:
            # The blocking acquire cannot be interrupted. If it already — or
            # eventually — succeeds with no owner left to release it, undo
            # the acquisition immediately so the resource is never wedged by
            # a cancelled waiter.
            future.add_done_callback(_release_abandoned_acquire(self._lock))
            raise

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
