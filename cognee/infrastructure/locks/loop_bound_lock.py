"""A module-safe async lock that binds to the running event loop on use.

A plain ``asyncio.Lock()`` created at import time lazily binds to the first
event loop that contends it (CPython 3.10+, via
``asyncio.mixins._LoopBoundMixin``) and afterwards raises
``RuntimeError: ... is bound to a different event loop`` if it is contended again
from a different running loop. Any process that drives cognee from more than one
event loop -- repeated ``asyncio.run(...)`` calls, a fresh loop per request,
``pytest-asyncio`` with function-scoped loops, or a multi-worker ingest -- can
therefore hit that error on cognee's module-level locks.

``LoopBoundLock`` keeps one real ``asyncio.Lock`` per running loop, created on
first use and stored in a ``WeakKeyDictionary`` keyed by the loop so locks for
closed loops are garbage-collected. Each loop gets its own lock, which is the
only thing an ``asyncio.Lock`` can do anyway -- it cannot synchronize across
loops -- so a module-level singleton was already incorrect for multi-loop
callers. Single-loop behavior is unchanged; the cross-loop failure is removed.
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from weakref import WeakKeyDictionary


class LoopBoundLock:
    """Drop-in replacement for a module-level ``asyncio.Lock``.

    Lazily instantiates one underlying ``asyncio.Lock`` per running event loop,
    so a single instance created at import time is safe to use from any number of
    event loops. Supports ``async with`` and the ``acquire`` / ``release`` subset
    of the ``asyncio.Lock`` interface that cognee uses.

    Use it via ``async with`` (or a paired ``acquire`` / ``release`` on the same
    loop). Like ``asyncio.Lock`` it serializes tasks within a single loop and does
    not synchronize across loops.
    """

    __slots__ = ("_locks",)

    def __init__(self) -> None:
        self._locks: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            WeakKeyDictionary()
        )

    def _lock_for_running_loop(self) -> asyncio.Lock:
        """Return this loop's underlying lock, creating it on first use."""
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop] = lock
        return lock

    async def acquire(self) -> bool:
        """Acquire the lock for the running event loop."""
        return await self._lock_for_running_loop().acquire()

    def release(self) -> None:
        """Release the running loop's lock; raise if it was not acquired here."""
        lock = self._locks.get(asyncio.get_running_loop())
        if lock is None:
            raise RuntimeError("release() called without a matching acquire() on this event loop")
        lock.release()

    async def __aenter__(self) -> "LoopBoundLock":
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
