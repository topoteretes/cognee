import pytest
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock

from cognee.modules.graph.methods import delete_dataset_related_edges


class DummyScalarResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class FakeEdge:
    def __init__(self, edge_id):
        self.id = edge_id


@pytest.mark.asyncio
async def test_delete_dataset_related_edges_deletes_found_rows():
    session = SimpleNamespace()
    session.scalars = AsyncMock(return_value=DummyScalarResult([FakeEdge(1), FakeEdge(2)]))
    session.execute = AsyncMock()

    await delete_dataset_related_edges(uuid4(), session=session)

    session.scalars.assert_awaited_once()
    session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_dataset_related_edges_handles_empty_list():
    session = SimpleNamespace()
    session.scalars = AsyncMock(return_value=DummyScalarResult([]))
    session.execute = AsyncMock()

    await delete_dataset_related_edges(uuid4(), session=session)

    session.scalars.assert_awaited_once()
    session.execute.assert_awaited_once()
