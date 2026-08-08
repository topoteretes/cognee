import asyncio
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.tasks.storage.index_data_points import index_data_points
from cognee.infrastructure.engine import DataPoint


class TestDataPoint(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


@pytest.mark.asyncio
async def test_index_data_points_calls_vector_engine():
    """Test that index_data_points creates vector index and indexes data."""
    data_points = [TestDataPoint(name="test1")]

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

    peak = await _run_and_measure_peak_concurrency(
        data_points, batch_size=2, max_concurrent_data_points=6
    )

    assert peak == 3


@pytest.mark.asyncio
async def test_concurrency_floors_at_one_when_batch_size_exceeds_limit():
    """batch_size > max_concurrent_data_points must still run one batch, not deadlock."""
    data_points = [TestDataPoint(name=f"point{i}") for i in range(10)]

    peak = await _run_and_measure_peak_concurrency(
        data_points, batch_size=100, max_concurrent_data_points=6
    )

    assert peak == 1
