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

    with patch.dict(
        index_data_points.__globals__,
        {"get_vector_engine": lambda: mock_vector_engine},
    ):
        await index_data_points(data_points)

    assert mock_vector_engine.create_vector_index.await_count >= 1
    assert mock_vector_engine.index_data_points.await_count >= 1
