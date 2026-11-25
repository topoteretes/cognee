"""Deleting a dataset removes its routing rows before its stores are dropped.

Dropping the dataset's graph and vector databases is the irreversible step. The
routing rows live in the default store, which can be unreachable, so they go
first: a failure there aborts the delete with the stores, rows and files still
in place, and the delete can be retried. Afterwards the ids the rows derive
from are gone with the ``Data`` rows, so nothing could address them again.
"""

import importlib
import types
from uuid import uuid4

import pytest

from cognee.modules.data.methods.delete_dataset import delete_dataset

delete_dataset_module = importlib.import_module("cognee.modules.data.methods.delete_dataset")
routing = importlib.import_module("cognee.modules.search.dataset_routing")


class _Rows:
    def __init__(self, ids):
        self._ids = ids

    def scalars(self):
        return self

    def all(self):
        return list(self._ids)


class _FakeSession:
    def __init__(self, dataset_database, data_ids, steps):
        self._dataset_database = dataset_database
        self._data_ids = data_ids
        self._steps = steps

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement):
        return _Rows(self._data_ids)

    async def scalar(self, statement):
        return self._dataset_database

    async def commit(self):
        pass

    async def get(self, model, dataset_id):
        return None


def _wire(monkeypatch, *, dataset_database, data_ids, steps, remove_routes):
    class FakeRelationalEngine:
        engine = types.SimpleNamespace(dialect=types.SimpleNamespace(name="postgresql"))

        def get_async_session(self):
            return _FakeSession(dataset_database, data_ids, steps)

        async def delete_data_entity(self, data_id, dataset_id):
            steps.append(("relational", data_id))

    class FakeHandler:
        def __init__(self, layer):
            self.layer = layer

        async def delete_dataset(self, row):
            steps.append((self.layer, row.id))

    monkeypatch.setattr(delete_dataset_module, "get_relational_engine", FakeRelationalEngine)
    monkeypatch.setattr(
        delete_dataset_module,
        "get_graph_dataset_database_handler",
        lambda row: {"handler_instance": FakeHandler("graph")},
    )
    monkeypatch.setattr(
        delete_dataset_module,
        "get_vector_dataset_database_handler",
        lambda row: {"handler_instance": FakeHandler("vector")},
    )
    monkeypatch.setattr(routing, "remove_dataset_routes", remove_routes)


@pytest.mark.asyncio
async def test_routing_rows_go_before_the_stores_and_the_rows(monkeypatch):
    dataset_id, data_id = uuid4(), uuid4()
    dataset_database = types.SimpleNamespace(id=uuid4(), dataset_id=dataset_id)
    steps = []

    async def remove_dataset_routes(routed_dataset_id, document_ids):
        steps.append(("routing", routed_dataset_id, list(document_ids)))

    _wire(
        monkeypatch,
        dataset_database=dataset_database,
        data_ids=[data_id],
        steps=steps,
        remove_routes=remove_dataset_routes,
    )

    await delete_dataset(types.SimpleNamespace(id=dataset_id))

    assert steps == [
        ("routing", dataset_id, [data_id]),
        ("graph", dataset_database.id),
        ("vector", dataset_database.id),
        ("relational", data_id),
    ]


@pytest.mark.asyncio
async def test_an_unreachable_routing_store_aborts_before_anything_is_dropped(monkeypatch):
    dataset_id, data_id = uuid4(), uuid4()
    dataset_database = types.SimpleNamespace(id=uuid4(), dataset_id=dataset_id)
    steps = []

    async def remove_dataset_routes(routed_dataset_id, document_ids):
        raise ConnectionError("default store down")

    _wire(
        monkeypatch,
        dataset_database=dataset_database,
        data_ids=[data_id],
        steps=steps,
        remove_routes=remove_dataset_routes,
    )

    with pytest.raises(ConnectionError):
        await delete_dataset(types.SimpleNamespace(id=dataset_id))

    assert steps == [], "the stores and the rows must still be intact so the delete can be retried"
