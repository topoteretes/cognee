import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage.index_data_points import index_data_points


class TestDataPoint(DataPoint):
    name: str
    description: str = "test description"


@pytest.mark.asyncio
async def test_index_data_points_calls_vector_engine():
    """Test that index_data_points creates vector index and indexes data."""
    data_points = [TestDataPoint(name="test1")]
    data_points[0].metadata["index_fields"] = ["name"]

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=100)

    async def _get_vector_engine():
        return mock_vector_engine

    with patch.dict(
        index_data_points.__globals__,
        {"get_vector_engine_async": _get_vector_engine},
    ):
        await index_data_points(data_points)

    assert mock_vector_engine.create_vector_index.await_count >= 1
    assert mock_vector_engine.index_data_points.await_count >= 1


async def _run_and_measure_peak_concurrency(data_points, batch_size, max_concurrent_data_points):
    """Run index_data_points and return the peak number of concurrent index calls."""
    in_flight = 0
    peak = 0

    async def _index(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=batch_size)
    mock_vector_engine.index_data_points = AsyncMock(side_effect=_index)

    with patch.dict(
        index_data_points.__globals__,
        {
            "get_embedding_context_config": lambda: SimpleNamespace(
                embedding_max_concurrent_data_points=max_concurrent_data_points
            )
        },
    ):
        await index_data_points(data_points, vector_engine=mock_vector_engine)

    return peak


@pytest.mark.asyncio
async def test_concurrency_derived_from_max_concurrent_data_points():
    """Concurrent batches = max_concurrent_data_points // batch_size (6 // 2 = 3)."""
    data_points = [TestDataPoint(name=f"point{i}") for i in range(20)]
    for data_point in data_points:
        data_point.metadata["index_fields"] = ["name"]

    peak = await _run_and_measure_peak_concurrency(
        data_points, batch_size=2, max_concurrent_data_points=6
    )

    assert peak == 3


@pytest.mark.asyncio
async def test_concurrency_floors_at_one_when_batch_size_exceeds_limit():
    """batch_size > max_concurrent_data_points must still run one batch, not deadlock."""
    data_points = [TestDataPoint(name=f"point{i}") for i in range(10)]
    for data_point in data_points:
        data_point.metadata["index_fields"] = ["name"]

    peak = await _run_and_measure_peak_concurrency(
        data_points, batch_size=100, max_concurrent_data_points=6
    )

    assert peak == 1


@pytest.mark.asyncio
async def test_index_data_points_does_not_mutate_metadata():
    data_point = TestDataPoint(name="test")
    data_point.metadata["index_fields"] = ["name", "description"]

    original_index_fields = list(data_point.metadata["index_fields"])

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=100)

    async def _get_vector_engine():
        return mock_vector_engine

    with patch.dict(
        index_data_points.__globals__,
        {"get_vector_engine_async": _get_vector_engine},
    ):
        await index_data_points([data_point])

    assert data_point.metadata["index_fields"] == original_index_fields


# --- skipping re-indexing of unchanged data points ---------------------------


def _engine_with_stored(stored_rows, *, retrieve_raises=False):
    """A vector engine whose store already holds ``stored_rows``."""
    engine = AsyncMock()
    engine.embedding_engine.get_batch_size = MagicMock(return_value=100)
    if retrieve_raises:
        engine.retrieve = AsyncMock(side_effect=RuntimeError("store unavailable"))
    else:
        engine.retrieve = AsyncMock(return_value=stored_rows)
    return engine


def _indexed_names(engine):
    """Names the engine was actually asked to index."""
    names = []
    for call in engine.index_data_points.await_args_list:
        for point in call.args[2]:
            names.append(point.name)
    return names


async def _run(points, engine, *, skip=True):
    for point in points:
        point.metadata["index_fields"] = ["name"]

    async def _get_vector_engine():
        return engine

    with patch.dict(
        index_data_points.__globals__,
        {
            "get_vector_engine_async": _get_vector_engine,
            "get_embedding_context_config": lambda: SimpleNamespace(
                embedding_max_concurrent_data_points=150,
                skip_unchanged_vector_writes=skip,
            ),
        },
    ):
        await index_data_points(points, vector_engine=engine)


@pytest.mark.asyncio
async def test_unchanged_point_is_not_reindexed():
    """The whole point: an identical value must not be re-embedded or rewritten."""
    point = TestDataPoint(name="unchanged")
    engine = _engine_with_stored(
        [SimpleNamespace(id=point.id, payload={"name": "unchanged"})]
    )
    await _run([point], engine)
    assert engine.index_data_points.await_count == 0


@pytest.mark.asyncio
async def test_changed_point_is_reindexed():
    point = TestDataPoint(name="new value")
    engine = _engine_with_stored(
        [SimpleNamespace(id=point.id, payload={"name": "old value"})]
    )
    await _run([point], engine)
    assert _indexed_names(engine) == ["new value"]


@pytest.mark.asyncio
async def test_point_the_store_does_not_know_is_indexed():
    point = TestDataPoint(name="brand new")
    engine = _engine_with_stored([])
    await _run([point], engine)
    assert _indexed_names(engine) == ["brand new"]


@pytest.mark.asyncio
async def test_payload_without_the_indexed_field_is_indexed():
    """Absent evidence is not evidence of sameness."""
    point = TestDataPoint(name="value")
    engine = _engine_with_stored(
        [SimpleNamespace(id=point.id, payload={"something_else": "value"})]
    )
    await _run([point], engine)
    assert _indexed_names(engine) == ["value"]


@pytest.mark.asyncio
async def test_retrieve_failure_indexes_everything():
    """Fail open. A store that cannot answer must never cause a skip."""
    point = TestDataPoint(name="value")
    engine = _engine_with_stored(None, retrieve_raises=True)
    await _run([point], engine)
    assert _indexed_names(engine) == ["value"]


@pytest.mark.asyncio
async def test_mixed_batch_indexes_only_the_changed_ones():
    same = TestDataPoint(name="same")
    different = TestDataPoint(name="different now")
    engine = _engine_with_stored(
        [
            SimpleNamespace(id=same.id, payload={"name": "same"}),
            SimpleNamespace(id=different.id, payload={"name": "different before"}),
        ]
    )
    await _run([same, different], engine)
    assert _indexed_names(engine) == ["different now"]


@pytest.mark.asyncio
async def test_disabling_the_flag_restores_unconditional_indexing():
    point = TestDataPoint(name="unchanged")
    engine = _engine_with_stored(
        [SimpleNamespace(id=point.id, payload={"name": "unchanged"})]
    )
    await _run([point], engine, skip=False)
    assert _indexed_names(engine) == ["unchanged"]
    assert engine.retrieve.await_count == 0, "the flag must skip the lookup entirely"
