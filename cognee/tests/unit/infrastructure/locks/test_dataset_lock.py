"""Tests for the shared per-dataset lock (cognee.infrastructure.locks.dataset_lock).

The lock registry is shared between pipeline runs (add/cognify/memify) and
delete operations, so a delete on a dataset must wait for an in-flight
pipeline run on that dataset (and vice versa), while operations on
different datasets proceed in parallel.
"""

import asyncio
import threading
from unittest.mock import patch
from uuid import uuid4

import pytest

from cognee.infrastructure.locks import DatasetLock, dataset_lock, get_dataset_lock, held_datasets


@pytest.mark.asyncio
async def test_same_dataset_operations_are_serialized():
    """Two concurrent dataset_lock blocks on the same dataset never overlap."""
    dataset_id = uuid4()
    events = []

    async def operation(name: str):
        async with dataset_lock(dataset_id):
            events.append(f"{name}:start")
            await asyncio.sleep(0.01)
            events.append(f"{name}:end")

    await asyncio.gather(operation("a"), operation("b"))

    # Whichever started first must have ended before the other started.
    assert events in (
        ["a:start", "a:end", "b:start", "b:end"],
        ["b:start", "b:end", "a:start", "a:end"],
    )


@pytest.mark.asyncio
async def test_different_datasets_run_in_parallel():
    """Locks are per-dataset: operations on different datasets interleave."""
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()

    async def hold_first():
        async with dataset_lock(uuid4()):
            first_entered.set()
            await asyncio.wait_for(second_entered.wait(), timeout=1)

    async def hold_second():
        await asyncio.wait_for(first_entered.wait(), timeout=1)
        async with dataset_lock(uuid4()):
            second_entered.set()

    # Would deadlock (and time out) if the second dataset waited on the first.
    await asyncio.gather(hold_first(), hold_second())


@pytest.mark.asyncio
async def test_reentrant_acquire_does_not_deadlock():
    """A nested operation on an already-held dataset takes the re-entrant path."""
    dataset_id = uuid4()

    async with dataset_lock(dataset_id):
        assert dataset_id in held_datasets.get()
        async with dataset_lock(dataset_id):
            pass  # would self-deadlock without re-entrancy

    assert dataset_id not in held_datasets.get()


@pytest.mark.asyncio
async def test_held_marker_propagates_to_child_tasks():
    """Child tasks spawned under the lock see the dataset as held (ContextVar copy),
    so e.g. a delete started from inside a pipeline task doesn't self-deadlock."""
    dataset_id = uuid4()

    async def nested_delete():
        async with dataset_lock(dataset_id):
            return dataset_id in held_datasets.get()

    async with dataset_lock(dataset_id):
        was_held = await asyncio.create_task(nested_delete())

    assert was_held


@pytest.mark.asyncio
async def test_delete_waits_for_pipeline_holding_the_lock():
    """dataset_lock shares its registry with the pipeline's get_dataset_lock:
    while a pipeline run holds the dataset's lock, a delete must wait."""
    dataset_id = uuid4()
    delete_ran = asyncio.Event()

    # Simulate an in-flight pipeline run (run_pipeline_per_dataset acquires
    # the lock through the same get_dataset_lock registry).
    pipeline_lock = await get_dataset_lock(dataset_id)
    await pipeline_lock.acquire()

    async def delete_operation():
        async with dataset_lock(dataset_id):
            delete_ran.set()

    delete_task = asyncio.create_task(delete_operation())
    await asyncio.sleep(0.01)
    assert not delete_ran.is_set(), "delete must wait while the pipeline holds the lock"

    pipeline_lock.release()
    await asyncio.wait_for(delete_task, timeout=1)
    assert delete_ran.is_set()


# ---------------------------------------------------------------------------
# Event-loop agnosticism: the cached per-dataset lock must work across event
# loops. asyncio.Lock binds to the first loop that awaits it, so the previous
# registry crashed with "is bound to a different event loop" for any embedder
# running cognee operations via asyncio.run on worker threads — and a lock
# left acquired by a finished loop wedged the dataset until process restart.
# ---------------------------------------------------------------------------


def test_lock_usable_across_sequential_event_loops():
    """Two asyncio.run calls (two loops) may use the same cached dataset lock."""
    dataset_id = uuid4()

    async def operation():
        async with dataset_lock(dataset_id):
            await asyncio.sleep(0)

    asyncio.run(operation())
    # Raised RuntimeError("... is bound to a different event loop") before.
    asyncio.run(operation())


def test_same_dataset_serialized_across_threads_with_own_loops():
    """Operations on one dataset from different threads/loops never overlap."""
    dataset_id = uuid4()
    events: list[str] = []
    first_holds = threading.Event()

    async def first():
        async with dataset_lock(dataset_id):
            events.append("a:start")
            first_holds.set()
            await asyncio.sleep(0.1)
            events.append("a:end")

    async def second():
        async with dataset_lock(dataset_id):
            events.append("b:start")
            events.append("b:end")

    thread_a = threading.Thread(target=lambda: asyncio.run(first()))
    thread_a.start()
    assert first_holds.wait(timeout=2)
    thread_b = threading.Thread(target=lambda: asyncio.run(second()))
    thread_b.start()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)

    assert events == ["a:start", "a:end", "b:start", "b:end"]


def test_different_datasets_do_not_block_across_loops():
    """A dataset lock held on one loop must not stall another dataset's lock."""
    held = threading.Event()
    release = threading.Event()

    async def hold_other_dataset():
        async with dataset_lock(uuid4()):
            held.set()
            await asyncio.to_thread(release.wait, 2)

    holder = threading.Thread(target=lambda: asyncio.run(hold_other_dataset()))
    holder.start()
    assert held.wait(timeout=2)
    try:

        async def use_own_dataset():
            async with dataset_lock(uuid4()):
                pass

        asyncio.run(use_own_dataset())  # must not wait on the held lock
    finally:
        release.set()
        holder.join(timeout=5)


# ---------------------------------------------------------------------------
# Lock-ordering guard (SDK-483): canonical order is dataset lock -> queue slot.
# A task already holding a DatasetQueue slot must not acquire a dataset lock.
# ---------------------------------------------------------------------------

GET_DATASET_QUEUE_SETTINGS = (
    "cognee.infrastructure.databases.dataset_queue.queue.get_dataset_queue_settings"
)


@pytest.fixture
def fresh_enabled_queue():
    """A fresh enabled DatasetQueue singleton, torn down after the test."""
    from cognee.infrastructure.databases.dataset_queue import dataset_queue

    dataset_queue._instance = None
    with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
        mock_settings.return_value.enabled = True
        mock_settings.return_value.max_concurrent = 2
        # An unset MagicMock attribute floats to 1.0, which would give the queue
        # a 1-second idle TTL — release_slot_for below would then start a REAL
        # reaper daemon that force-closes every subprocess engine idle >1s for
        # the rest of the pytest process (the "LanceDBAdapter is closed" flake
        # in unrelated tests). 0 keeps the release path reaper-free, same as
        # TestReleaseSlotFor.
        mock_settings.return_value.idle_ttl_seconds = 0
        yield dataset_queue()
    dataset_queue._instance = None


@pytest.mark.asyncio
async def test_lock_after_slot_on_same_dataset_raises(fresh_enabled_queue):
    """Slot -> lock on the same dataset is the SDK-483 deadlock; it must raise."""
    dataset_id = uuid4()
    await fresh_enabled_queue.ensure_slot(dataset_id)
    try:
        with pytest.raises(RuntimeError, match="Lock-order inversion"):
            await get_dataset_lock(dataset_id)
    finally:
        await fresh_enabled_queue.release_slot_for(dataset_id)


@pytest.mark.asyncio
async def test_lock_after_slot_on_other_dataset_warns_but_proceeds(fresh_enabled_queue):
    """A slot for another dataset is a cross-dataset risk: warn, don't raise."""
    slot_dataset_id, lock_dataset_id = uuid4(), uuid4()
    await fresh_enabled_queue.ensure_slot(slot_dataset_id)
    try:
        lock = await get_dataset_lock(lock_dataset_id)
        assert isinstance(lock, DatasetLock)
    finally:
        await fresh_enabled_queue.release_slot_for(slot_dataset_id)


@pytest.mark.asyncio
async def test_canonical_order_lock_then_slot_is_allowed(fresh_enabled_queue):
    """The canonical order (lock first, then slot) must stay unrestricted."""
    dataset_id = uuid4()
    async with dataset_lock(dataset_id):
        await fresh_enabled_queue.ensure_slot(dataset_id)
        await fresh_enabled_queue.release_slot_for(dataset_id)


@pytest.mark.asyncio
async def test_lock_without_any_slot_is_unaffected(fresh_enabled_queue):
    """The guard is inert for tasks holding no slot."""
    lock = await get_dataset_lock(uuid4())
    assert isinstance(lock, DatasetLock)
