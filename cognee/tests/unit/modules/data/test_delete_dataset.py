"""Tests for delete_dataset's isolated-storage teardown (Finding 15, COG-6335 review).

delete_dataset used to inline the same two graph/vector handler.delete_dataset()
calls that delete_isolated_dataset_storage already factors out for the
memory-only reset path. This module must reuse that primitive instead of
duplicating it.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

module = importlib.import_module("cognee.modules.data.methods.delete_dataset")

pytestmark = pytest.mark.asyncio


class _FakeScalarsResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class _FakeExecuteResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return _FakeScalarsResult(self._items)


class _FakeSession:
    def __init__(self, dataset_database, data_ids=(), dataset_row=None):
        self._dataset_database = dataset_database
        self._data_ids = list(data_ids)
        self._dataset_row = dataset_row
        self.commits = 0
        self.deleted = []

    async def execute(self, _statement):
        # Covers both the sqlite PRAGMA text() call (return value unused) and
        # select(Data.id)...scalars().all().
        return _FakeExecuteResult(self._data_ids)

    async def scalar(self, _statement):
        return self._dataset_database

    async def get(self, _model, _id):
        return self._dataset_row

    async def delete(self, row):
        self.deleted.append(row)

    async def commit(self):
        self.commits += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeEngine:
    def __init__(self, sessions):
        self._sessions = list(sessions)
        self.engine = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
        self.delete_data_entity = AsyncMock()

    def get_async_session(self):
        return self._sessions.pop(0)


async def test_delete_dataset_reuses_delete_isolated_dataset_storage(monkeypatch):
    dataset_database = SimpleNamespace(
        graph_dataset_database_handler="ladybug",
        vector_dataset_database_handler="lancedb",
    )
    dataset = SimpleNamespace(id="dataset-id")
    session1 = _FakeSession(dataset_database, data_ids=[])
    session2 = _FakeSession(dataset_database, dataset_row=dataset)
    engine = _FakeEngine([session1, session2])

    reset_mock = AsyncMock()
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset_mock)
    monkeypatch.setattr(module, "get_relational_engine", lambda: engine)

    await module.delete_dataset(dataset)

    reset_mock.assert_awaited_once_with(dataset_database)


async def test_delete_dataset_skips_isolated_storage_reset_when_no_dataset_database_row(
    monkeypatch,
):
    dataset = SimpleNamespace(id="dataset-id")
    session1 = _FakeSession(dataset_database=None, data_ids=[])
    session2 = _FakeSession(dataset_database=None, dataset_row=dataset)
    engine = _FakeEngine([session1, session2])

    reset_mock = AsyncMock()
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset_mock)
    monkeypatch.setattr(module, "get_relational_engine", lambda: engine)

    await module.delete_dataset(dataset)

    reset_mock.assert_not_called()


async def test_delete_dataset_module_no_longer_imports_the_inline_handler_getters():
    """The two handler-getter imports this module used to inline are now
    unused -- delete_isolated_dataset_storage owns that call."""
    assert not hasattr(module, "get_graph_dataset_database_handler")
    assert not hasattr(module, "get_vector_dataset_database_handler")
