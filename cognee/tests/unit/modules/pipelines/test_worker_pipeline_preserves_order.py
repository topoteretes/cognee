"""Order-preservation regression test for the worker pipeline executor.

When every stage uses ``FixedWorkers(num_workers=1)``, a multi-stage pipeline
must yield results in the same order as the inputs were pushed.
"""

import asyncio

import pytest

from cognee.modules.pipelines.operations.worker_pipeline import (
    FixedWorkers,
    _ErroredItem,
    run_worker_pipeline,
)
from cognee.modules.pipelines.tasks.task import Task


async def _identity(value):
    return value


async def _plus_one(value):
    # Jitter to give the asyncio scheduler reasons to interleave across stages.
    await asyncio.sleep(0)
    return value + 1


async def _times_two(value):
    await asyncio.sleep(0)
    return value * 2


@pytest.mark.asyncio
async def test_single_worker_preserves_order():
    tasks = [
        Task(_identity, workers=FixedWorkers(1)),
        Task(_plus_one, workers=FixedWorkers(1)),
        Task(_times_two, workers=FixedWorkers(1)),
    ]
    inputs = list(range(100))
    expected = [(i + 1) * 2 for i in inputs]

    out = []
    async for envelope in run_worker_pipeline(
        tasks=tasks,
        data_iterable=inputs,
        user=None,
        ctx=None,
        data_per_batch=8,
    ):
        assert not isinstance(envelope.value, _ErroredItem)
        out.append(envelope.value)

    assert out == expected, f"Expected ordered output but got {out!r}"


def test_single_worker_preserves_order_sync():
    asyncio.run(test_single_worker_preserves_order())


if __name__ == "__main__":
    test_single_worker_preserves_order_sync()
