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
            ),
            # One write per embedding request, so the batch count is the request count.
            "get_vectordb_config": lambda: SimpleNamespace(vector_write_batch_size=batch_size),
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
async def test_write_batches_hold_vector_write_batch_size_rows():
    """The write batch is VECTOR_WRITE_BATCH_SIZE rows, not the embedding request size:
    300 points with 32-text requests and 128-row writes are three writes."""
    data_points = [TestDataPoint(name=f"point{i}") for i in range(300)]
    for data_point in data_points:
        data_point.metadata["index_fields"] = ["name"]
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=32)

    with patch.dict(
        index_data_points.__globals__,
        {"get_vectordb_config": lambda: SimpleNamespace(vector_write_batch_size=128)},
    ):
        await index_data_points(data_points, vector_engine=mock_vector_engine)

    batch_sizes = [
        len(call.args[2]) for call in mock_vector_engine.index_data_points.await_args_list
    ]
    assert sorted(batch_sizes, reverse=True) == [128, 128, 44]


@pytest.mark.asyncio
async def test_a_point_with_one_index_field_is_handed_over_without_a_copy():
    """Adapters embed the first index field, so nothing has to change on the point; only
    a point indexed on several fields gets a copy per field, each naming its field."""

    class TwoFieldPoint(DataPoint):
        name: str
        description: str
        metadata: dict = {"index_fields": ["name", "description"]}

    single = TestDataPoint(name="single")
    single.metadata["index_fields"] = ["name"]
    double = TwoFieldPoint(name="n", description="d")
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=100)

    await index_data_points([single, double], vector_engine=mock_vector_engine)

    handed_over = {
        (call.args[0], call.args[1]): call.args[2]
        for call in mock_vector_engine.index_data_points.await_args_list
    }
    assert handed_over[("TestDataPoint", "name")][0] is single
    (by_name,) = handed_over[("TwoFieldPoint", "name")]
    (by_description,) = handed_over[("TwoFieldPoint", "description")]
    assert by_name is not double and by_description is not double
    assert by_name.metadata["index_fields"] == ["name"]
    assert by_description.metadata["index_fields"] == ["description"]
    assert double.metadata["index_fields"] == ["name", "description"]


@pytest.mark.asyncio
async def test_a_shared_semaphore_bounds_concurrent_calls_together():
    """Two index calls handed one semaphore never exceed its bound between them."""
    in_flight = 0
    peak = 0

    async def _index(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=1)
    mock_vector_engine.index_data_points = AsyncMock(side_effect=_index)
    first = [TestDataPoint(name=f"a{i}") for i in range(6)]
    second = [TestDataPoint(name=f"b{i}") for i in range(6)]
    for data_point in first + second:
        data_point.metadata["index_fields"] = ["name"]
    shared = asyncio.Semaphore(2)

    with patch.dict(
        index_data_points.__globals__,
        {"get_vectordb_config": lambda: SimpleNamespace(vector_write_batch_size=1)},
    ):
        await asyncio.gather(
            index_data_points(first, vector_engine=mock_vector_engine, semaphore=shared),
            index_data_points(second, vector_engine=mock_vector_engine, semaphore=shared),
        )

    assert peak == 2


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
