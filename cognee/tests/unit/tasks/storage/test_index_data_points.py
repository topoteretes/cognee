import pytest
from unittest.mock import AsyncMock, MagicMock

from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage.index_data_points import index_data_points


class MultiFieldPoint(DataPoint):
    problem: str = ""
    conclusion: str = ""
    metadata: dict = {"index_fields": ["problem", "conclusion"]}


@pytest.mark.asyncio
async def test_index_data_points_indexes_every_field_without_mutating_original():
    point = MultiFieldPoint(problem="PROBLEM_AAA", conclusion="CONCLUSION_BBB")
    vector_engine = MagicMock()
    vector_engine.create_vector_index = AsyncMock()
    vector_engine.index_data_points = AsyncMock()
    vector_engine.embedding_engine.get_batch_size.return_value = 10

    await index_data_points([point], vector_engine=vector_engine)

    indexed_field_names = {
        call.args[1] for call in vector_engine.index_data_points.await_args_list
    }
    assert indexed_field_names == {"problem", "conclusion"}
    assert point.metadata["index_fields"] == ["problem", "conclusion"]
