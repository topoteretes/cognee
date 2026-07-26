"""Tests for ``create_dataset`` concurrent-creation safety.

Concurrent calls for the same dataset name race between the existence SELECT
and the INSERT and, because the dataset id is deterministic, collide on the
primary key. ``create_dataset`` must tolerate losing that race (catch the
IntegrityError and return the winner's committed row) without weakening
isolation: the recovered row must satisfy the same name/owner/tenant-scoped
query as the original lookup, so an id collision with any other row —
another owner's or tenant's dataset — stays an error.
"""

import asyncio
import sys
from types import SimpleNamespace
from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest
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


def _matches(row, statement):
    """Evaluate a select statement's WHERE criteria against a row."""
    whereclause = statement.whereclause
    for clause in getattr(whereclause, "clauses", [whereclause]):
        column_name = clause.left.name
        expected = getattr(clause.right, "value", None)  # IS NULL has no value
        if getattr(row, column_name) != expected:
            return False
    return True


class _FakeSession:
    """Emulates the SELECT -> INSERT window against a shared rows-by-id store.

    Every await yields control so concurrent tasks interleave exactly like
    real DB round-trips. SELECTs honor the statement's WHERE criteria, and
    commit enforces the primary-key unique constraint the way the real
    database does: inserting a duplicate id raises IntegrityError.
    """

    def __init__(self, rows_by_id):
        self._rows_by_id = rows_by_id
        self._pending_add = None

    async def scalars(self, statement):
        await asyncio.sleep(0)
        return _FakeScalarResult(
            next((row for row in self._rows_by_id.values() if _matches(row, statement)), None)
        )

    def add(self, dataset):
        self._pending_add = dataset

    async def commit(self):
        await asyncio.sleep(0)
        if self._pending_add is not None:
            row, self._pending_add = self._pending_add, None
            if row.id in self._rows_by_id:
                raise IntegrityError("duplicate key value", None, Exception())
            self._rows_by_id[row.id] = row

    async def rollback(self):
        await asyncio.sleep(0)
        self._pending_add = None


class _CollideAndVanishSession(_FakeSession):
    """Insert collides, but the winner's row is gone by the re-select."""

    async def scalars(self, _statement):
        await asyncio.sleep(0)
        return _FakeScalarResult(None)

    async def commit(self):
        await asyncio.sleep(0)
        raise IntegrityError("duplicate key value", None, Exception())


def _install_deterministic_dataset_id(monkeypatch):
    async def fake_get_unique_dataset_id(dataset_name, user):
        return uuid5(NAMESPACE_OID, f"{dataset_name}{user.id}")

    monkeypatch.setattr(create_dataset_module, "get_unique_dataset_id", fake_get_unique_dataset_id)


@pytest.mark.asyncio
async def test_concurrent_same_dataset_creation_does_not_raise(monkeypatch):
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    _install_deterministic_dataset_id(monkeypatch)

    rows_by_id = {}
    datasets = await asyncio.gather(
        *[create_dataset("race_dataset", user, session=_FakeSession(rows_by_id)) for _ in range(10)]
    )

    assert len(rows_by_id) == 1
    assert len({dataset.id for dataset in datasets}) == 1


@pytest.mark.asyncio
async def test_colliding_foreign_dataset_is_not_returned(monkeypatch):
    """An id collision with another owner's dataset must error, never leak the row."""
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    _install_deterministic_dataset_id(monkeypatch)

    dataset_id = uuid5(NAMESPACE_OID, f"race_dataset{user.id}")
    foreign_dataset = SimpleNamespace(
        id=dataset_id, name="foreign_dataset", owner_id=uuid4(), tenant_id=uuid4()
    )
    rows_by_id = {dataset_id: foreign_dataset}

    with pytest.raises(IntegrityError):
        await create_dataset("race_dataset", user, session=_FakeSession(rows_by_id))

    assert rows_by_id == {dataset_id: foreign_dataset}


@pytest.mark.asyncio
async def test_vanished_winner_row_reraises_integrity_error(monkeypatch):
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    _install_deterministic_dataset_id(monkeypatch)

    with pytest.raises(IntegrityError):
        await create_dataset("race_dataset", user, session=_CollideAndVanishSession({}))
