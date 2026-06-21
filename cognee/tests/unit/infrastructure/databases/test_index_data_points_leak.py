import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.tasks.storage.index_data_points import index_data_points
from cognee.infrastructure.engine import DataPoint

class MultiFieldDataPoint(DataPoint):
    name: str = "Test Entity"
    description: str = "Test Description"
    metadata: dict = {"index_fields": ["name", "description"]}

@pytest.mark.asyncio
async def test_index_data_points_does_not_overwrite_metadata():
    data_point = MultiFieldDataPoint()
    data_points = [data_point]

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=100)

    # Patch get_vector_engine to return our mock
    with patch(
        "cognee.tasks.storage.index_data_points.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        await index_data_points(data_points)

    # Assert that the original object's metadata was not corrupted/mutated
    assert data_point.metadata["index_fields"] == ["name", "description"], (
        f"Original metadata index_fields was mutated to: {data_point.metadata['index_fields']}"
    )

    # Verify that the vector indexer calls index_data_points for both fields with correct copied points
    # Let's inspect the calls to vector_engine.index_data_points.
    # index_data_points signature: index_data_points(type_name, field_name, batch)
    calls = mock_vector_engine.index_data_points.call_args_list
    assert len(calls) == 2, f"Expected 2 index calls, got {len(calls)}"

    first_call_field = calls[0][0][1]
    first_call_batch = calls[0][0][2]
    second_call_field = calls[1][0][1]
    second_call_batch = calls[1][0][2]

    # Verify field names and their copied data point metadata
    assert first_call_field == "name"
    assert first_call_batch[0].metadata["index_fields"] == ["name"], (
        f"Expected first batch data points to have index_fields=['name'], got {first_call_batch[0].metadata['index_fields']}"
    )

    assert second_call_field == "description"
    assert second_call_batch[0].metadata["index_fields"] == ["description"], (
        f"Expected second batch data points to have index_fields=['description'], got {second_call_batch[0].metadata['index_fields']}"
    )
