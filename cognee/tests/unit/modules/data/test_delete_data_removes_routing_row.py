"""Deleting a document's row removes its dataset routing rows first.

The bucket row's id derives from the ``Data`` row, so once that row is gone
nothing can address the routing row again: it has to go before the relational
delete, on the one path every document deletion shares. The index nodes the
rows mirror, the document's bucket and the dataset's root (its text names the
document), go the same way.
"""

import importlib
import types
from uuid import uuid4

import pytest

from cognee.modules.data.methods.delete_data import delete_data

delete_data_module = importlib.import_module("cognee.modules.data.methods.delete_data")
routing = importlib.import_module("cognee.modules.search.dataset_routing")
index = importlib.import_module("cognee.tasks.memify.global_context_index")


@pytest.mark.asyncio
async def test_delete_data_removes_the_routing_row_before_the_relational_row(monkeypatch):
    dataset_id, data_id = uuid4(), uuid4()
    steps = []

    async def remove_dataset_routes(routed_dataset_id, document_ids):
        steps.append(("routing", routed_dataset_id, list(document_ids)))

    class FakeRelationalEngine:
        async def delete_data_entity(self, deleted_id, deleted_dataset_id):
            steps.append(("relational", deleted_dataset_id, [deleted_id]))
            return "receipt"

    async def remove_document_index(indexed_dataset_id, indexed_data_id):
        steps.append(("index", indexed_dataset_id, indexed_data_id))

    monkeypatch.setattr(routing, "remove_dataset_routes", remove_dataset_routes)
    monkeypatch.setattr(index, "remove_document_index", remove_document_index)
    monkeypatch.setattr(delete_data_module, "get_relational_engine", FakeRelationalEngine)

    result = await delete_data(types.SimpleNamespace(id=data_id, __tablename__="data"), dataset_id)

    assert result == "receipt"
    assert steps == [
        ("routing", dataset_id, [data_id]),
        ("index", dataset_id, data_id),
        ("relational", dataset_id, [data_id]),
    ]
