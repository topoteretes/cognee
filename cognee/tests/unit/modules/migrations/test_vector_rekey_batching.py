"""Batching contract of the migrations' shared vector re-embed path.

``index_data_points_batched`` must mirror cognify's own write path
(``cognee/tasks/storage/index_data_points.py``): embedding-engine-sized
batches, concurrency bounded by the process-wide batch slots
(``EMBEDDING_MAX_CONCURRENT_REQUESTS``).
One unbatched call over a whole collection is what providers reject
(OpenAI caps an embeddings request at 300k tokens total).
"""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import cognee.infrastructure.databases.vector.embeddings.config as embedding_config_module
from cognee.modules.migrations.versions._vector_rekey import (
    RekeyedPoint,
    index_data_points_batched,
)
from cognee.shared import rate_limiting


def _make_points(count: int) -> list:
    return [RekeyedPoint(text=f"text {i}") for i in range(count)]


def _make_engine(batch_size: int) -> AsyncMock:
    engine = AsyncMock()
    engine.embedding_engine.get_batch_size = MagicMock(return_value=batch_size)
    return engine


@contextmanager
def _config_patch(max_concurrent_requests: int):
    """Set EMBEDDING_MAX_CONCURRENT_REQUESTS, with fresh process-wide slots."""
    config = SimpleNamespace(embedding_max_concurrent_requests=max_concurrent_requests)
    with (
        patch.object(embedding_config_module, "get_embedding_config", lambda: config),
        patch.object(rate_limiting, "_concurrency_state", {}),
    ):
        yield


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
async def test_first_batch_settles_before_others_start():
    """The first batch runs alone so collection creation / lazy schema
    migration settles before concurrent upserts (a concurrent get_collection
    during LanceDB's mismatch-triggered table rewrite raises
    CollectionNotFoundError)."""
    events = []

    async def _index(*args, **kwargs):
        events.append("start")
        await asyncio.sleep(0)
        events.append("end")

    engine = _make_engine(batch_size=2)
    engine.index_data_points = AsyncMock(side_effect=_index)

    with _config_patch(150):
        await index_data_points_batched(engine, "DocumentChunk", "text", _make_points(10))

    assert engine.index_data_points.await_count == 5
    # First call fully completes before any other call starts.
    assert events[0] == "start" and events[1] == "end"


@pytest.mark.asyncio
async def test_concurrent_batches_bounded_by_max_concurrent_requests():
    """Concurrent batches = EMBEDDING_MAX_CONCURRENT_REQUESTS."""
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

    with _config_patch(3):
        await index_data_points_batched(engine, "EdgeType", "relationship_name", _make_points(20))

    assert peak == 3


@pytest.mark.asyncio
async def test_batch_size_above_the_cap_still_runs():
    """A batch larger than EMBEDDING_MAX_CONCURRENT_REQUESTS still runs, not deadlocks."""
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
