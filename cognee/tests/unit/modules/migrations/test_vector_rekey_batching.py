"""Batching contract of the migrations' shared vector re-embed path.

``index_data_points_batched`` must mirror cognify's own write path
(``cognee/tasks/storage/index_data_points.py``): embedding-engine-sized
batches, concurrency bounded by ``embedding_max_concurrent_data_points``.
One unbatched call over a whole collection is what providers reject
(OpenAI caps an embeddings request at 300k tokens total).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.migrations.versions._vector_rekey import (
    RekeyedPoint,
    index_data_points_batched,
)


def _make_points(count: int) -> list:
    return [RekeyedPoint(text=f"text {i}") for i in range(count)]


def _make_engine(batch_size: int) -> AsyncMock:
    engine = AsyncMock()
    engine.embedding_engine.get_batch_size = MagicMock(return_value=batch_size)
    return engine


def _config_patch(max_concurrent_data_points: int):
    return patch.dict(
        index_data_points_batched.__globals__,
        {
            "get_embedding_context_config": lambda: SimpleNamespace(
                embedding_max_concurrent_data_points=max_concurrent_data_points
            )
        },
    )


@pytest.mark.asyncio
async def test_splits_into_batches_of_engine_batch_size():
    points = _make_points(25)
    engine = _make_engine(batch_size=10)

    with _config_patch(150):
        await index_data_points_batched(engine, "DocumentChunk", "text", points)

    calls = engine.index_data_points.await_args_list
    assert len(calls) == 3
    batches = [call.args[2] for call in calls]
    assert [len(batch) for batch in batches] == [10, 10, 5]
    # Every point indexed exactly once, under the caller's collection kind/field.
    indexed = [point for batch in batches for point in batch]
    assert [point.text for point in indexed] == [point.text for point in points]
    assert all(call.args[:2] == ("DocumentChunk", "text") for call in calls)


@pytest.mark.asyncio
async def test_concurrency_derived_from_max_concurrent_data_points():
    """Concurrent batches = max_concurrent_data_points // batch_size (6 // 2 = 3)."""
    in_flight = 0
    peak = 0

    async def _index(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1

    engine = _make_engine(batch_size=2)
    engine.index_data_points = AsyncMock(side_effect=_index)

    with _config_patch(6):
        await index_data_points_batched(engine, "EdgeType", "relationship_name", _make_points(20))

    assert peak == 3


@pytest.mark.asyncio
async def test_concurrency_floors_at_one_when_batch_size_exceeds_limit():
    """batch_size > max_concurrent_data_points must still run one batch, not deadlock."""
    engine = _make_engine(batch_size=100)

    with _config_patch(10):
        await index_data_points_batched(engine, "Triplet", "text", _make_points(5))

    assert engine.index_data_points.await_count == 1


@pytest.mark.asyncio
async def test_no_points_makes_no_calls():
    engine = _make_engine(batch_size=10)

    with _config_patch(150):
        await index_data_points_batched(engine, "DocumentChunk", "text", [])

    engine.index_data_points.assert_not_awaited()
    engine.embedding_engine.get_batch_size.assert_not_called()
