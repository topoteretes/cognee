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


class MultiFieldDataPoint(DataPoint):
    """DataPoint with multiple index_fields — used to verify the shallow-copy fix."""

    problem: str = ""
    conclusion: str = ""
    follow_up: str = ""
    metadata: dict = {"index_fields": ["problem", "conclusion", "follow_up"]}


@pytest.mark.asyncio
async def test_index_data_points_multiple_fields_all_indexed():
    data_point = MultiFieldDataPoint(
        problem="PROBLEM_AAA", conclusion="CONCLUSION_BBB", follow_up="FOLLOWUP_CCC"
    )
    data_points = [data_point]

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=100)

    with patch.dict(
        index_data_points.__globals__,
        {"get_vector_engine": lambda: mock_vector_engine},
    ):
        await index_data_points(data_points)

    indexed_calls = mock_vector_engine.index_data_points.await_args_list
    indexed_by_field = {call.args[1] for call in indexed_calls}
    assert indexed_by_field == {"problem", "conclusion", "follow_up"}

    field_to_text = {}
    for call in indexed_calls:
        field_name = call.args[1]
        batch = call.args[2]
        assert len(batch) == 1
        indexed_dp = batch[0]
        assert indexed_dp.metadata["index_fields"] == [field_name]
        field_to_text[field_name] = getattr(indexed_dp, field_name)

    assert field_to_text == {
        "problem": "PROBLEM_AAA",
        "conclusion": "CONCLUSION_BBB",
        "follow_up": "FOLLOWUP_CCC",
    }

    assert data_point.metadata["index_fields"] == [
        "problem",
        "conclusion",
        "follow_up",
    ]
