"""The dataset-queue budget bounds DISTINCT datasets, not (task, dataset) pairs.

Regression tests for the cloud add/search hang (#4673): concurrent requests to
the *same* dataset used to each take their own semaphore permit, so a single hot
dataset hit by ``max_concurrent`` requests exhausted the whole budget and wedged
every later ``/add`` and ``/search`` — even though those requests share ONE
engine. The permit is now owned per dataset: the first holder acquires it, the
last holder releases it.

Engine teardown (``_release_subprocess_engines``) is patched out: these tests
exercise the permit accounting, which is orthogonal to subprocess lifecycle.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue


def _queue(max_concurrent: int) -> DatasetQueue:
    q = DatasetQueue(enabled=True, max_concurrent=max_concurrent, idle_ttl_seconds=0.0)
    # Orthogonal to permit accounting, and needs no real DB context.
    q._release_subprocess_engines = MagicMock()
    return q


async def _hold_same_dataset(q: DatasetQueue, n: int, ds: str):
    """Spawn ``n`` tasks that all hold ``ds`` at once, then release together."""
    entered = [asyncio.Event() for _ in range(n)]
    release = asyncio.Event()

    async def worker(i: int) -> None:
        await q.ensure_slot(ds)
        entered[i].set()
        await release.wait()
        await q.release_slot_for(ds)

    tasks = [asyncio.create_task(worker(i)) for i in range(n)]
    # Before the fix, only ``max_concurrent`` workers ever enter and this
    # times out; after it, all ``n`` share one permit and enter immediately.
    await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered)), timeout=3.0)
    value_while_held = q._semaphore._value
    active = q.active_dataset_ids()
    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=3.0)
    return value_while_held, active


def test_many_concurrent_requests_to_one_dataset_take_one_permit():
    """N concurrent holders of the SAME dataset consume exactly one budget slot."""

    async def body():
        q = _queue(max_concurrent=2)
        # 5 concurrent requests, budget of 2: pre-fix this wedges at the 3rd.
        value_while_held, active = await _hold_same_dataset(q, n=5, ds="hot")
        assert value_while_held == 1, (
            f"one dataset must hold ONE permit; budget=2 → value should be 1, got {value_while_held}"
        )
        assert active == {"hot"}
        # Everything released: full budget restored, last holder tore down once.
        assert q._semaphore._value == 2
        assert q.active_dataset_ids() == set()
        q._release_subprocess_engines.assert_called_once()

    asyncio.run(body())


def test_distinct_datasets_still_bounded_by_budget():
    """The fix must not remove the bound: DISTINCT datasets still cap at budget.

    Each dataset is held by a live worker (a bare ``ensure_slot`` task would
    finish and the task-end backstop would free its permit at once).
    """

    async def body():
        q = _queue(max_concurrent=2)
        entered = {ds: asyncio.Event() for ds in ("d0", "d1", "d2")}
        release = {ds: asyncio.Event() for ds in ("d0", "d1", "d2")}

        async def worker(ds: str):
            await q.ensure_slot(ds)
            entered[ds].set()
            await release[ds].wait()
            await q.release_slot_for(ds)

        t0 = asyncio.create_task(worker("d0"))
        t1 = asyncio.create_task(worker("d1"))
        await asyncio.wait_for(
            asyncio.gather(entered["d0"].wait(), entered["d1"].wait()), timeout=2.0
        )
        assert q._semaphore._value == 0, "two distinct datasets must consume both slots"

        t2 = asyncio.create_task(worker("d2"))
        await asyncio.sleep(0.2)
        assert not entered["d2"].is_set(), "a 3rd distinct dataset must block on a full budget"

        release["d0"].set()  # free one slot
        await asyncio.wait_for(entered["d2"].wait(), timeout=2.0)  # d2 now proceeds
        assert q._semaphore._value == 0

        release["d1"].set()
        release["d2"].set()
        await asyncio.wait_for(asyncio.gather(t0, t1, t2), timeout=2.0)
        assert q._semaphore._value == 2

    asyncio.run(body())


def test_last_holder_releases_not_the_first():
    """Releasing all-but-one holder keeps the permit; the last one frees it."""

    async def body():
        q = _queue(max_concurrent=3)
        holders = [asyncio.Event() for _ in range(3)]
        go = asyncio.Event()
        step = asyncio.Queue()

        async def worker(i: int):
            await q.ensure_slot("shared")
            holders[i].set()
            await go.wait()
            await step.get()  # released one at a time by the test
            await q.release_slot_for("shared")

        tasks = [asyncio.create_task(worker(i)) for i in range(3)]
        await asyncio.wait_for(asyncio.gather(*(h.wait() for h in holders)), timeout=3.0)
        assert q._semaphore._value == 2  # one permit for three holders
        go.set()

        step.put_nowait(None)  # release holder 0
        await asyncio.sleep(0.1)
        assert q._semaphore._value == 2, "permit must survive while holders remain"
        q._release_subprocess_engines.assert_not_called()

        step.put_nowait(None)  # release holder 1
        await asyncio.sleep(0.1)
        assert q._semaphore._value == 2

        step.put_nowait(None)  # release holder 2 — the last
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=3.0)
        assert q._semaphore._value == 3, "last holder frees the permit"
        q._release_subprocess_engines.assert_called_once()

    asyncio.run(body())


def test_same_task_reentrancy_uses_one_permit():
    """Nested ensure_slot for one task+dataset is depth-counted, not re-acquired."""

    async def body():
        q = _queue(max_concurrent=2)
        await q.ensure_slot("D")
        await q.ensure_slot("D")  # re-entrant
        assert q._semaphore._value == 1
        await q.release_slot_for("D")  # depth 2 -> 1, still held
        assert q._semaphore._value == 1
        q._release_subprocess_engines.assert_not_called()
        await q.release_slot_for("D")  # depth 1 -> 0, freed
        assert q._semaphore._value == 2
        q._release_subprocess_engines.assert_called_once()

    asyncio.run(body())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
