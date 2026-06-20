"""Tests that get_dataset_data eagerly loads the datasets relationship to avoid N+1 queries."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.methods.get_last_added_data import get_last_added_data


@dataclass
class FakeDataset:
    id: uuid.UUID
    name: str = "test-dataset"


@dataclass
class FakeData:
    id: uuid.UUID
    name: str = "doc.txt"
    extension: str = "txt"
    mime_type: str = "text/plain"
    raw_data_location: str = "/tmp/doc.txt"
    created_at: object = None
    updated_at: object = None
    data_size: int = 100
    datasets: list = field(default_factory=list)


@pytest.mark.asyncio
async def test_get_dataset_data_uses_selectinload_and_prefetches_datasets():
    """get_dataset_data should populate .datasets inside the session so that
    accessing the relationship on a detached object does not fire additional
    queries (N+1 pattern)."""

    dataset_id = uuid.uuid4()
    data_id_1 = uuid.uuid4()
    data_id_2 = uuid.uuid4()

    data_item_1 = FakeData(id=data_id_1)
    data_item_2 = FakeData(id=data_id_2)
    data_items = [data_item_1, data_item_2]

    mock_scalars_result = MagicMock()
    mock_scalars_result.all.return_value = data_items

    mock_execute_result = MagicMock()
    mock_execute_result.scalars.return_value = mock_scalars_result

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_execute_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.get_async_session.return_value = mock_session

    with patch(
        "cognee.modules.data.methods.get_dataset_data.get_relational_engine",
        return_value=mock_engine,
    ):
        result = await get_dataset_data(dataset_id=dataset_id)

    assert mock_session.execute.call_count == 1
    assert len(result) == 2
    assert result[0].id == data_id_1
    assert result[1].id == data_id_2


@pytest.mark.asyncio
async def test_get_last_added_data_prefetches_datasets():
    """get_last_added_data should also eagerly load .datasets to prevent N+1."""

    dataset_id = uuid.uuid4()
    data_id = uuid.uuid4()

    data_item = FakeData(id=data_id)

    mock_execute_result = MagicMock()
    mock_execute_result.scalar_one_or_none.return_value = data_item

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_execute_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.get_async_session.return_value = mock_session

    with patch(
        "cognee.modules.data.methods.get_last_added_data.get_relational_engine",
        return_value=mock_engine,
    ):
        result = await get_last_added_data(dataset_id=dataset_id)

    assert mock_session.execute.call_count == 1
    assert result is not None
    assert result.id == data_id
