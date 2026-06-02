import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.tasks.storage.index_data_points import index_data_points
from cognee.infrastructure.engine import DataPoint


class TestDataPoint(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class MultiFieldDataPoint(DataPoint):
    title: str
    summary: str
    body: str
    metadata: dict = {"index_fields": ["title", "summary", "body"]}


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


@pytest.mark.asyncio
async def test_index_data_points_all_fields_indexed_independently():
    """Regression test for shallow-copy metadata mutation bug.

    When a DataPoint has multiple index_fields, each field must be indexed
    into its own vector collection with a copy that only lists that field.
    The shared-metadata shallow-copy bug caused all copies to end up with
    the last field's name, so only one field was ever truly indexed.
    """
    dp = MultiFieldDataPoint(title="T", summary="S", body="B")

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=100)

    indexed_calls = []

    async def capture_index(type_name, field_name, batch):
        for point in batch:
            indexed_calls.append((field_name, point.metadata["index_fields"]))

    mock_vector_engine.index_data_points.side_effect = capture_index

    with patch.dict(
        index_data_points.__globals__,
        {"get_vector_engine": lambda: mock_vector_engine},
    ):
        await index_data_points([dp])

    indexed_field_names = [field for field, _ in indexed_calls]
    assert sorted(indexed_field_names) == ["body", "summary", "title"], (
        f"Expected all three fields to be indexed, got: {indexed_field_names}"
    )

    for field_name, index_fields in indexed_calls:
        assert index_fields == [field_name], (
            f"Metadata for field '{field_name}' had index_fields={index_fields!r}; "
            f"expected [{field_name!r}]. Shallow-copy mutation bug still present."
        )
