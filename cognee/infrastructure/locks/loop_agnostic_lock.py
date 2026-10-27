"""Event-loop-agnostic async synchronization primitives.

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

:class:`LoopAgnosticLock` runs ``asyncio.Lock``'s own algorithm — a FIFO
queue of waiter futures, the head woken on each release, newcomers never
jumping past a live waiter — with two changes: the state is guarded by a
``threading.Lock`` so it can be used from any loop or thread, and a waiter
is woken through its *own* loop's ``call_soon_threadsafe`` so the holder may
release from a different loop or from no loop at all. No thread is ever
parked on behalf of a waiter: an uncontended acquire is one mutex
round-trip, and a contended one costs one scheduled callback, the same as
``asyncio.Lock``'s ``call_soon``. There is no pool to exhaust, so any number
of waiters on one lock cannot delay another lock.

A waiter that can no longer be woken — its loop was closed while it was
parked — is dropped, and the live waiter behind it woken, by whoever
touches the lock next: the next ``release()``, the next ``acquire()`` on a
free lock (a release's wake-up is only confirmed when the woken waiter
claims; an acquire that finds the lock free re-issues it to the current
head), and, so that a lone parked waiter recovers even when nobody comes
by, each parked waiter's own periodic re-check. A loop that went away
mid-wait therefore never keeps the lock, and nothing is left to block
interpreter exit. (A loop that is stopped but never closed, and never run
again, is indistinguishable from a slow one; its waiter keeps its place
like an ``asyncio.Lock`` waiter on a stalled loop would.)

:class:`LoopAgnosticEvent` is the matching ``asyncio.Event``: ``set()`` may
be called from any thread and wakes the waiters of every loop.
"""

import asyncio
import threading
from collections import deque

# How often a parked waiter re-checks whether the head it queued behind was
# orphaned by a closed loop (see the module docstring). Only parked waiters
# pay for this, one timer each; it does not affect the uncontended path.
_RECHECK_INTERVAL_SECONDS = 1.0


def _resolve(future: asyncio.Future) -> None:
    """Runs on the waiter's loop: hand it the wake-up unless it already left."""
    if not future.done():
        future.set_result(True)


def _schedule_wake(future: asyncio.Future) -> bool:
    """Schedule ``future`` to be resolved on its own loop; False if that loop is gone."""
    loop = future.get_loop()
    if loop.is_closed():
        return False
    try:
        loop.call_soon_threadsafe(_resolve, future)
    except RuntimeError:  # closed between the check and the call
        return False
    return True


def _is_dead(future: asyncio.Future) -> bool:
    """A waiter that will never claim: cancelled, or its loop closed under it."""
    return future.cancelled() or future.get_loop().is_closed()


class LoopAgnosticLock:
    """Async mutual exclusion usable from any event loop or thread.

    Mirrors the slice of the ``asyncio.Lock`` interface the codebase uses
    (``async with``, awaitable ``acquire()``, ``release()``, ``locked()``),
    with ``asyncio.Lock``'s FIFO fairness: once a waiter is queued, a newcomer
    cannot take the lock ahead of it.
    """

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._locked = False
        self._waiters: deque[asyncio.Future] = deque()

    def locked(self) -> bool:
        return self._locked

    async def acquire(self) -> bool:
        loop = asyncio.get_running_loop()
        with self._mutex:
            if not self._locked:
                # The lock is free, so a wake-up is in flight to the head
                # waiter (or there is none). Re-issue it: this drops a head
                # whose loop died since the release and wakes the live one
                # behind it, so a lost wake-up never strands the queue.
                self._wake_next_locked()
                if not self._waiters:
                    self._locked = True
                    return True
            future = loop.create_future()
            self._waiters.append(future)

        while True:
            try:
                # Bounded wait: a parked waiter re-checks now and then in case
                # the head it queued behind was orphaned by a closed loop
                # while no newcomer came by to notice (see module docstring).
                done, _ = await asyncio.wait({future}, timeout=_RECHECK_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                with self._mutex:
                    self._discard_locked(future)
                    if not self._locked:
                        # A release may have picked us just as we were
                        # cancelled; pass the wake-up on so it is not lost.
                        self._wake_next_locked()
                raise

            with self._mutex:
                if not done:
                    if not self._locked:
                        self._wake_next_locked()  # may wake us, if we are head now
                    continue
                self._discard_locked(future)
                if not self._locked:
                    self._locked = True
                    return True
                # Defensive: woken while someone else holds the lock (cannot
                # happen with the invariants above). Keep our place at the
                # head of the queue rather than barge or give up.
                future = loop.create_future()
                self._waiters.appendleft(future)

    def release(self) -> None:
        with self._mutex:
            if not self._locked:
                raise RuntimeError("Lock is not acquired.")
            self._locked = False
            self._wake_next_locked()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    # -- internals; the caller holds ``self._mutex`` ------------------------

    def _discard_locked(self, future: asyncio.Future) -> None:
        try:
            self._waiters.remove(future)
        except ValueError:
            pass

    def _wake_next_locked(self) -> None:
        while self._waiters:
            head = self._waiters[0]
            if _is_dead(head):
                self._waiters.popleft()
                continue
            if head.done():
                # Already woken; it claims the lock when its task resumes.
                return
            if _schedule_wake(head):
                return
            self._waiters.popleft()


class LoopAgnosticEvent:
    """``asyncio.Event`` whose ``set()`` works from any thread or loop.

    Waiters on several loops may wait at once; ``set()`` wakes all of them
    through their own loops. ``clear()`` and ``is_set()`` behave as on
    ``asyncio.Event``.
    """

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._flag = False
        self._waiters: list[asyncio.Future] = []

    def is_set(self) -> bool:
        return self._flag

    def set(self) -> None:
        with self._mutex:
            if self._flag:
                return
            self._flag = True
            waiters, self._waiters = self._waiters, []
        for future in waiters:
            _schedule_wake(future)  # a closed loop's waiter is simply gone

    def clear(self) -> None:
        with self._mutex:
            self._flag = False

    async def wait(self) -> bool:
        loop = asyncio.get_running_loop()
        with self._mutex:
            if self._flag:
                return True
            future = loop.create_future()
            self._waiters.append(future)
        try:
            await future
        except asyncio.CancelledError:
            with self._mutex:
                if future in self._waiters:
                    self._waiters.remove(future)
            raise
        return True
