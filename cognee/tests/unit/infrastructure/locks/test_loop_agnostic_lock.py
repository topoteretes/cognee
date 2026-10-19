"""Tests for the loop-agnostic primitives (cognee.infrastructure.locks.loop_agnostic_lock).

``LoopAgnosticLock`` must keep ``asyncio.Lock``'s guarantees (FIFO fairness,
cancellation safety) while working across event loops and threads, without
parking a thread per waiter and without letting a dead loop's waiter keep
the lock. ``LoopAgnosticEvent`` must be settable from any thread.
"""

import asyncio
import threading

import pytest

from cognee.infrastructure.locks.loop_agnostic_lock import LoopAgnosticEvent, LoopAgnosticLock


@pytest.mark.asyncio
async def test_newcomer_does_not_barge_past_parked_waiter():
    """FIFO: a release followed by an immediate re-acquire queues behind the waiter."""
    lock = LoopAgnosticLock()
    order: list[str] = []
    await lock.acquire()

    async def waiter():
        async with lock:
            order.append("waiter")

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)  # parked
    lock.release()
    # Newcomer before the parked waiter has resumed: must not jump the queue.
    async with lock:
        order.append("newcomer")
    await task

    assert order == ["waiter", "newcomer"]


@pytest.mark.asyncio
async def test_many_waiters_are_served_in_order_and_do_not_block_other_locks():
    """No pool to exhaust: 50 parked waiters on one lock leave another lock free."""
    busy, other = LoopAgnosticLock(), LoopAgnosticLock()
    served: list[int] = []
    await busy.acquire()

    async def waiter(n: int):
        async with busy:
            served.append(n)

    tasks = []
    for n in range(50):
        tasks.append(asyncio.create_task(waiter(n)))
        await asyncio.sleep(0)  # deterministic enqueue order

    # Would deadlock with a per-waiter thread pool of 32.
    await asyncio.wait_for(other.acquire(), timeout=1)
    other.release()

    busy.release()
    await asyncio.gather(*tasks)
    assert served == list(range(50))


@pytest.mark.asyncio
async def test_cancelled_waiter_passes_the_wake_up_on():
    """Cancelling the waiter a release just picked must not strand the next one."""
    lock = LoopAgnosticLock()
    await lock.acquire()

    async def waiter():
        await lock.acquire()
        lock.release()

    first = asyncio.create_task(waiter())
    second = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)

    lock.release()  # wake-up scheduled for ``first``
    first.cancel()  # ...which gives it up before resuming
    with pytest.raises(asyncio.CancelledError):
        await first

    await asyncio.wait_for(second, timeout=1)
    assert not lock.locked()


def test_release_from_outside_any_loop_and_across_loops():
    """The holder may release on another loop, or with no loop running at all."""
    lock = LoopAgnosticLock()

    asyncio.run(lock.acquire())
    assert lock.locked()
    lock.release()  # no running loop here

    async def waiter():
        await asyncio.wait_for(lock.acquire(), timeout=1)
        lock.release()

    asyncio.run(lock.acquire())
    holder_release = threading.Timer(0.05, lock.release)
    holder_release.start()
    asyncio.run(waiter())  # a third loop, woken by a plain thread
    holder_release.join()


def test_waiter_whose_loop_closed_is_skipped_on_release():
    """A waiter abandoned by its closed loop must not inherit (and keep) the lock."""
    lock = LoopAgnosticLock()
    asyncio.run(lock.acquire())

    dead_loop = asyncio.new_event_loop()
    abandoned = dead_loop.create_task(lock.acquire())
    dead_loop.run_until_complete(asyncio.sleep(0.01))  # parked
    dead_loop.close()  # closed with the waiter still parked, never cancelled

    lock.release()  # the dead waiter must be passed over

    async def newcomer():
        await asyncio.wait_for(lock.acquire(), timeout=1)
        lock.release()

    asyncio.run(newcomer())
    assert not lock.locked()
    del abandoned


def test_waiter_woken_then_orphaned_by_loop_close_does_not_keep_lock():
    """Release before the close: the scheduled wake-up never runs; newcomers proceed."""
    lock = LoopAgnosticLock()
    asyncio.run(lock.acquire())

    dead_loop = asyncio.new_event_loop()
    abandoned = dead_loop.create_task(lock.acquire())
    dead_loop.run_until_complete(asyncio.sleep(0.01))
    lock.release()  # wake-up scheduled on ``dead_loop``...
    dead_loop.close()  # ...which never runs it

    async def newcomer():
        await asyncio.wait_for(lock.acquire(), timeout=1)
        lock.release()

    asyncio.run(newcomer())
    assert not lock.locked()
    del abandoned


def test_release_of_unlocked_lock_raises():
    with pytest.raises(RuntimeError):
        LoopAgnosticLock().release()


@pytest.mark.asyncio
async def test_event_set_from_another_thread_wakes_waiter():
    event = LoopAgnosticEvent()
    assert not event.is_set()
    threading.Timer(0.05, event.set).start()
    assert await asyncio.wait_for(event.wait(), timeout=1)
    assert event.is_set()


def test_event_wakes_waiters_on_several_loops():
    event = LoopAgnosticEvent()
    woke = []

    def wait_on_own_loop(name: str):
        async def go():
            await asyncio.wait_for(event.wait(), timeout=2)
            woke.append(name)

        asyncio.run(go())

    threads = [threading.Thread(target=wait_on_own_loop, args=(n,)) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    threading.Event().wait(0.05)
    event.set()
    for thread in threads:
        thread.join(timeout=5)
    assert sorted(woke) == ["a", "b"]

    event.clear()
    assert not event.is_set()

    async def blocks_after_clear():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(event.wait(), timeout=0.05)

    asyncio.run(blocks_after_clear())
