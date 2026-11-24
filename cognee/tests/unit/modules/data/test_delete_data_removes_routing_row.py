"""Deleting a document's row removes its dataset routing rows first.

The bucket row's id derives from the ``Data`` row, so once that row is gone
nothing can address the routing row again: it has to go before the relational
delete, on the one path every document deletion shares. The index nodes the
rows mirror, the document's bucket and the dataset's root (its text names the
document), go the same way. Once the rows are gone the root is written again
from the documents that remain, so a broad query keeps routing to the dataset.
"""

import importlib
import types
from uuid import uuid4

import pytest

from cognee.modules.data.methods.delete_data import delete_data

delete_data_module = importlib.import_module("cognee.modules.data.methods.delete_data")
routing = importlib.import_module("cognee.modules.search.dataset_routing")
index = importlib.import_module("cognee.tasks.memify.global_context_index")

USER = types.SimpleNamespace(id=uuid4())


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

    async def remove_document_index(indexed_dataset_id, indexed_data_ids):
        steps.append(("index", indexed_dataset_id, list(indexed_data_ids)))
        return True

    async def rebuild_document_index_root(rebuilt_dataset_id, user):
        steps.append(("rebuild", rebuilt_dataset_id, user))

    monkeypatch.setattr(routing, "remove_dataset_routes", remove_dataset_routes)
    monkeypatch.setattr(index, "remove_document_index", remove_document_index)
    monkeypatch.setattr(index, "rebuild_document_index_root", rebuild_document_index_root)
    monkeypatch.setattr(delete_data_module, "get_relational_engine", FakeRelationalEngine)

    await delete_data([types.SimpleNamespace(id=data_id, __tablename__="data")], dataset_id, USER)

    # The root is rebuilt only once the rows are gone: it is computed from the
    # dataset's Data rows, so an earlier rebuild would still name the document.
    assert steps == [
        ("routing", dataset_id, [data_id]),
        ("index", dataset_id, [data_id]),
        ("relational", dataset_id, [data_id]),
        ("rebuild", dataset_id, USER),
    ]


@pytest.mark.asyncio
async def test_deleting_many_rows_removes_their_index_and_routing_rows_once(monkeypatch):
    """A connector sweep hands every orphan over at once: one routing delete, one
    index removal (the root goes once), then the rows, then one root rebuild."""
    dataset_id, first, second = uuid4(), uuid4(), uuid4()
    steps = []

    async def remove_dataset_routes(routed_dataset_id, document_ids):
        steps.append(("routing", list(document_ids)))

    async def remove_document_index(indexed_dataset_id, indexed_data_ids):
        steps.append(("index", list(indexed_data_ids)))
        return True

    async def rebuild_document_index_root(rebuilt_dataset_id, user):
        steps.append(("rebuild", rebuilt_dataset_id))

    class FakeRelationalEngine:
        async def delete_data_entity(self, deleted_id, deleted_dataset_id):
            steps.append(("relational", [deleted_id]))

    monkeypatch.setattr(routing, "remove_dataset_routes", remove_dataset_routes)
    monkeypatch.setattr(index, "remove_document_index", remove_document_index)
    monkeypatch.setattr(index, "rebuild_document_index_root", rebuild_document_index_root)
    monkeypatch.setattr(delete_data_module, "get_relational_engine", FakeRelationalEngine)

    await delete_data(
        [
            types.SimpleNamespace(id=first, __tablename__="data"),
            types.SimpleNamespace(id=second, __tablename__="data"),
        ],
        dataset_id,
        USER,
    )

    assert steps == [
        ("routing", [first, second]),
        ("index", [first, second]),
        ("relational", [first]),
        ("relational", [second]),
        ("rebuild", dataset_id),
    ]


@pytest.mark.asyncio
async def test_a_dataset_that_had_no_index_gets_no_root(monkeypatch):
    """The removal found nothing to remove: the dataset was never indexed (or
    opted out of the stage), and writing a root would make routing treat it as
    indexed. The deletion itself is unchanged."""
    dataset_id, data_id = uuid4(), uuid4()
    steps = []

    async def remove_dataset_routes(routed_dataset_id, document_ids):
        steps.append("routing")

    async def remove_document_index(indexed_dataset_id, indexed_data_ids):
        steps.append("index")
        return False

    async def rebuild_document_index_root(rebuilt_dataset_id, user):
        steps.append("rebuild")

    class FakeRelationalEngine:
        async def delete_data_entity(self, deleted_id, deleted_dataset_id):
            steps.append("relational")

    monkeypatch.setattr(routing, "remove_dataset_routes", remove_dataset_routes)
    monkeypatch.setattr(index, "remove_document_index", remove_document_index)
    monkeypatch.setattr(index, "rebuild_document_index_root", rebuild_document_index_root)
    monkeypatch.setattr(delete_data_module, "get_relational_engine", FakeRelationalEngine)

    await delete_data([types.SimpleNamespace(id=data_id, __tablename__="data")], dataset_id, USER)

    assert steps == ["routing", "index", "relational"]
