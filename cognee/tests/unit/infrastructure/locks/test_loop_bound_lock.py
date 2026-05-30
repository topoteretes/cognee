"""Unit tests for the loop-bound async lock helper."""

import asyncio
import gc

import pytest

from cognee.infrastructure.locks.loop_bound_lock import LoopBoundLock


@pytest.mark.asyncio
async def test_serializes_concurrent_tasks_in_one_loop():
    """Within a single loop the lock gives mutual exclusion: two tasks cannot be
    inside the critical section at the same time."""
    lock = LoopBoundLock()
    events: list[str] = []

    async def worker(name: str, hold: float):
        async with lock:
            events.append(f"enter-{name}")
            await asyncio.sleep(hold)
            events.append(f"exit-{name}")

    await asyncio.gather(worker("a", 0.02), worker("b", 0.0))

    # Each enter is immediately followed by its matching exit (no overlap),
    # whichever task won the lock first.
    assert events in (
        ["enter-a", "exit-a", "enter-b", "exit-b"],
        ["enter-b", "exit-b", "enter-a", "exit-a"],
    )


def test_reused_across_event_loops_without_runtime_error():
    """A LoopBoundLock built once (e.g. at module scope) must work from more than
    one event loop. A plain module-level ``asyncio.Lock`` raises "bound to a
    different event loop" here once it has been contended on the first loop."""
    lock = LoopBoundLock()

    async def contend_once():
        async def hold():
            async with lock:
                await asyncio.sleep(0.01)

        # Two tasks contend so the contended (slow) acquire path runs -- that is
        # the path that performs asyncio's loop-binding check.
        await asyncio.gather(hold(), hold())

    asyncio.run(contend_once())  # binds the underlying lock to loop #1
    asyncio.run(contend_once())  # loop #2: must not raise


def test_underlying_lock_is_released_when_loop_is_gone():
    """Locks for closed loops are not retained (no per-loop leak)."""
    lock = LoopBoundLock()

    async def use():
        async with lock:
            pass

    asyncio.run(use())
    gc.collect()

    assert len(lock._locks) == 0


def test_each_loop_gets_an_independent_lock():
    """Two event loops must not serialize against each other: holding the lock
    in one loop does not block a different loop from acquiring it. A single
    shared lock would make loop B block on loop A here."""
    lock = LoopBoundLock()

    loop_a = asyncio.new_event_loop()
    try:
        loop_a.run_until_complete(lock.acquire())  # held in loop A, never released

        async def acquire_in_loop_b():
            # Independent per-loop lock -> acquires immediately; a shared lock
            # would block until the (never released) loop-A hold timed out.
            await asyncio.wait_for(lock.acquire(), timeout=1.0)
            lock.release()

        asyncio.run(acquire_in_loop_b())
    finally:
        loop_a.close()


def test_release_without_acquire_raises_without_creating_a_lock():
    """A release() with no matching acquire() raises and does not leave a stray
    lock behind."""
    lock = LoopBoundLock()

    async def bad_release():
        with pytest.raises(RuntimeError):
            lock.release()
        return len(lock._locks)

    assert asyncio.run(bad_release()) == 0
