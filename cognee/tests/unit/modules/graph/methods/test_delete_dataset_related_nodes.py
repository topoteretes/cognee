import pytest
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock

from cognee.modules.graph.methods import delete_dataset_related_nodes


class DummyScalarResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class FakeNode:
    def __init__(self, node_id):
        self.id = node_id


@pytest.mark.asyncio
async def test_delete_dataset_related_nodes_deletes_found_rows():
    session = SimpleNamespace()
    session.scalars = AsyncMock(return_value=DummyScalarResult([FakeNode(1), FakeNode(2)]))
    session.execute = AsyncMock()

    await delete_dataset_related_nodes(uuid4(), session=session)

    session.scalars.assert_awaited_once()
    session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_dataset_related_nodes_handles_empty_list():
    session = SimpleNamespace()
    session.scalars = AsyncMock(return_value=DummyScalarResult([]))
    session.execute = AsyncMock()

    await delete_dataset_related_nodes(uuid4(), session=session)

    session.scalars.assert_awaited_once()
    session.execute.assert_awaited_once()

