"""The improve claim queues: overlapping improves run one after another, like
pipeline runs on the per-dataset lock."""

import asyncio

import pytest

from cognee.infrastructure.locks.session_lock import (
    acquire_improve_lock_many,
    release_improve_lock_many,
)


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_run_sharing_a_key_waits_for_the_holder():
    await acquire_improve_lock_many(["session:u:a", "session:u:c"])
    waiter = asyncio.create_task(acquire_improve_lock_many(["session:u:b", "session:u:c"]))
    await _settle()
    assert not waiter.done()

    await release_improve_lock_many(["session:u:a", "session:u:c"])
    await asyncio.wait_for(waiter, 1)
    await release_improve_lock_many(["session:u:b", "session:u:c"])


@pytest.mark.asyncio
async def test_runs_sharing_no_key_do_not_wait():
    await acquire_improve_lock_many(["session:u:a", "session:u:c1"])
    await asyncio.wait_for(acquire_improve_lock_many(["session:u:b", "session:u:c2"]), 1)
    await release_improve_lock_many(["session:u:a", "session:u:c1"])
    await release_improve_lock_many(["session:u:b", "session:u:c2"])


@pytest.mark.asyncio
async def test_waiters_run_in_arrival_order():
    keys = ["session:u:c"]
    await acquire_improve_lock_many(keys)
    order = []

    async def run(name):
        await acquire_improve_lock_many(keys)
        order.append(name)
        await release_improve_lock_many(keys)

    tasks = []
    for name in ("first", "second", "third"):
        tasks.append(asyncio.create_task(run(name)))
        await _settle()
    await release_improve_lock_many(keys)
    await asyncio.wait_for(asyncio.gather(*tasks), 1)
    assert order == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_key_order_cannot_deadlock_two_runs():
    """Keys are taken in sorted order whatever order the caller lists them in."""

    async def run(keys):
        await acquire_improve_lock_many(keys)
        await _settle()
        await release_improve_lock_many(keys)

    await asyncio.wait_for(
        asyncio.gather(
            run(["session:u:a", "session:u:c"]),
            run(["session:u:c", "session:u:a"]),
        ),
        1,
    )


@pytest.mark.asyncio
async def test_a_cancelled_wait_releases_the_keys_it_already_took():
    await acquire_improve_lock_many(["session:u:z"])
    # "session:u:c" sorts first: it is taken, then the wait on "session:u:z" blocks.
    waiter = asyncio.create_task(acquire_improve_lock_many(["session:u:c", "session:u:z"]))
    await _settle()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    await asyncio.wait_for(acquire_improve_lock_many(["session:u:c"]), 1)
    await release_improve_lock_many(["session:u:c"])
    await release_improve_lock_many(["session:u:z"])


@pytest.mark.asyncio
async def test_releasing_a_key_that_is_not_held_raises():
    await acquire_improve_lock_many(["session:u:c"])
    await release_improve_lock_many(["session:u:c"])
    with pytest.raises(RuntimeError):
        await release_improve_lock_many(["session:u:c"])


@pytest.mark.asyncio
async def test_empty_keys_are_noops():
    await acquire_improve_lock_many([])
    await acquire_improve_lock_many(["", None])
    await release_improve_lock_many([])


def test_the_claim_can_be_used_from_more_than_one_event_loop():
    async def contended():
        await acquire_improve_lock_many(["session:u:c"])
        waiter = asyncio.create_task(acquire_improve_lock_many(["session:u:c"]))
        await _settle()
        await release_improve_lock_many(["session:u:c"])
        await asyncio.wait_for(waiter, 1)
        await release_improve_lock_many(["session:u:c"])

    asyncio.run(contended())
    asyncio.run(contended())
