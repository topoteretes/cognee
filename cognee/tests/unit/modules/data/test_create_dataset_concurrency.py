"""Tests for ``create_dataset`` concurrent-creation safety.

Concurrent calls for the same dataset name race between the existence SELECT
and the INSERT and, because the dataset id is deterministic, collide on the
primary key. ``create_dataset`` must tolerate losing that race (it catches the
IntegrityError and returns the winner's committed row) so no caller errors and
every caller gets the same dataset.
"""

import asyncio
import sys
from types import SimpleNamespace
from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError

from cognee.modules.data.methods import create_dataset

# The package __init__ rebinds the ``create_dataset`` attribute to the
# function, so the module itself must come from sys.modules.
create_dataset_module = sys.modules["cognee.modules.data.methods.create_dataset"]


class _FakeScalarResult:
    def __init__(self, value):
        self._value = value

    def first(self):
        return self._value


class _FakeSession:
    """Emulates the SELECT -> INSERT window against a shared row store.

    Every await yields control so concurrent tasks interleave exactly like
    real DB round-trips. Commit enforces the primary-key unique constraint the
    way the real database does: a duplicate insert raises IntegrityError
    unless the statement carries ON CONFLICT DO NOTHING.
    """

    bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

    def __init__(self, store):
        self._store = store
        self._pending_insert = None
        self._pending_add = None

    async def scalars(self, _query):
        await asyncio.sleep(0)
        return _FakeScalarResult(self._store.get("row"))

    async def execute(self, statement):
        await asyncio.sleep(0)
        compiled = statement.compile(dialect=sqlite.dialect())
        self._pending_insert = (dict(compiled.params), "ON CONFLICT" in str(compiled))

    def add(self, dataset):
        self._pending_add = dataset

    async def commit(self):
        await asyncio.sleep(0)

        if self._pending_add is not None:
            row, self._pending_add = self._pending_add, None
            self._apply_insert(row, handles_conflict=False)

        if self._pending_insert is not None:
            (values, handles_conflict), self._pending_insert = self._pending_insert, None
            self._apply_insert(SimpleNamespace(**values), handles_conflict)

    async def rollback(self):
        await asyncio.sleep(0)
        self._pending_insert = None
        self._pending_add = None

    def _apply_insert(self, row, handles_conflict):
        if self._store.get("row") is not None:
            if handles_conflict:
                return
            raise IntegrityError("duplicate key value", None, Exception())
        self._store["row"] = row
        self._store["insert_count"] = self._store.get("insert_count", 0) + 1


class _CollideAndVanishSession(_FakeSession):
    """Insert collides, but the winner's row is gone by the re-select."""

    async def scalars(self, _query):
        await asyncio.sleep(0)
        return _FakeScalarResult(None)

    async def commit(self):
        await asyncio.sleep(0)
        raise IntegrityError("duplicate key value", None, Exception())


@pytest.mark.asyncio
async def test_concurrent_same_dataset_creation_does_not_raise(monkeypatch):
    user = SimpleNamespace(id=uuid4(), tenant_id=None)

    async def fake_get_unique_dataset_id(dataset_name, user):
        return uuid5(NAMESPACE_OID, f"{dataset_name}{user.id}")

    monkeypatch.setattr(create_dataset_module, "get_unique_dataset_id", fake_get_unique_dataset_id)

    store = {}
    datasets = await asyncio.gather(
        *[create_dataset("race_dataset", user, session=_FakeSession(store)) for _ in range(10)]
    )

    assert store["insert_count"] == 1
    assert len({dataset.id for dataset in datasets}) == 1


@pytest.mark.asyncio
async def test_vanished_winner_row_reraises_integrity_error(monkeypatch):
    user = SimpleNamespace(id=uuid4(), tenant_id=None)

    async def fake_get_unique_dataset_id(dataset_name, user):
        return uuid5(NAMESPACE_OID, f"{dataset_name}{user.id}")

    monkeypatch.setattr(create_dataset_module, "get_unique_dataset_id", fake_get_unique_dataset_id)

    with pytest.raises(IntegrityError):
        await create_dataset("race_dataset", user, session=_CollideAndVanishSession({}))
