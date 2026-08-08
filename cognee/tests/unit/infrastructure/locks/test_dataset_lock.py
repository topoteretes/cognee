"""Tests for the shared per-dataset lock (cognee.infrastructure.locks.dataset_lock).

The lock registry is shared between pipeline runs (add/cognify/memify) and
delete operations, so a delete on a dataset must wait for an in-flight
pipeline run on that dataset (and vice versa), while operations on
different datasets proceed in parallel.
"""

import asyncio
from uuid import uuid4

import pytest

from cognee.infrastructure.locks import dataset_lock, get_dataset_lock, held_datasets


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
