"""pgvector retries a write that Postgres aborted to break a deadlock, and nothing else.

SQLAlchemy's asyncpg dialect raises its own ``DBAPIError``; the asyncpg
``DeadlockDetectedError`` is the cause of that error's ``orig``. A retry condition that
matched the asyncpg class never fired. These tests pin the predicate and the retry on
the two writes that still take the collection lock, ``delete_data_points`` and the
per-table detag of ``remove_belongs_to_set_tags``, with a session that deadlocks once.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from sqlalchemy import Column, MetaData, String, Table, text
from sqlalchemy.exc import DBAPIError
from tenacity import RetryError, wait_none

pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

from asyncpg import DeadlockDetectedError, UniqueViolationError

from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
    PGVectorAdapter,
    _is_deadlock,
)


def _wrapped(cause: Exception) -> DBAPIError:
    """The error SQLAlchemy raises for an asyncpg failure: a DBAPIError over the adapted error."""
    adapted = Exception("adapted asyncpg error")
    adapted.__cause__ = cause
    return DBAPIError("statement", {}, adapted)


class _Rows:
    def all(self):
        return [("row-1",), ("row-2",)]


class _Session:
    """Fails each execute with the queued errors first, then succeeds."""

    def __init__(self, errors: list[Exception], executed: list):
        self.errors = errors
        self.executed = executed

    async def execute(self, statement, params=None):
        if self.errors:
            raise self.errors.pop(0)
        self.executed.append(statement)
        return _Rows()

    async def commit(self):
        pass


def _adapter(errors: list[Exception], executed: list) -> PGVectorAdapter:
    adapter = object.__new__(PGVectorAdapter)
    adapter._write_locks = {}
    adapter._write_locks_guard = threading.Lock()
    table = Table("rows", MetaData(), Column("id", String))

    async def has_collection(name):
        return True

    async def get_table(name):
        return table

    @asynccontextmanager
    async def get_async_session():
        yield _Session(errors, executed)

    adapter.has_collection = has_collection
    adapter.get_table = get_table
    adapter.get_async_session = get_async_session
    return adapter


def test_is_deadlock_matches_only_the_wrapped_deadlock():
    assert _is_deadlock(_wrapped(DeadlockDetectedError("deadlock detected")))
    assert not _is_deadlock(_wrapped(UniqueViolationError("duplicate key")))
    assert not _is_deadlock(DeadlockDetectedError("deadlock detected"))
    assert not _is_deadlock(RuntimeError("deadlock detected"))


@pytest.mark.asyncio
async def test_delete_data_points_retries_a_deadlock():
    executed = []
    adapter = _adapter([_wrapped(DeadlockDetectedError("deadlock detected"))], executed)
    delete = PGVectorAdapter.delete_data_points.retry_with(wait=wait_none())

    await delete(adapter, "Entity_name", [uuid4()])

    assert len(executed) == 1


@pytest.mark.asyncio
async def test_delete_data_points_does_not_retry_other_errors():
    executed = []
    adapter = _adapter([_wrapped(UniqueViolationError("duplicate key"))], executed)
    delete = PGVectorAdapter.delete_data_points.retry_with(wait=wait_none())

    with pytest.raises(DBAPIError):
        await delete(adapter, "Entity_name", [uuid4()])

    assert executed == []


@pytest.mark.asyncio
async def test_delete_data_points_gives_up_after_three_deadlocks():
    deadlocks = [_wrapped(DeadlockDetectedError("deadlock detected")) for _ in range(3)]
    adapter = _adapter(deadlocks, [])
    delete = PGVectorAdapter.delete_data_points.retry_with(wait=wait_none())

    with pytest.raises(RetryError):
        await delete(adapter, "Entity_name", [uuid4()])


@pytest.mark.asyncio
async def test_remove_tags_from_table_retries_a_deadlock():
    executed = []
    adapter = _adapter([_wrapped(DeadlockDetectedError("deadlock detected"))], executed)
    detag = PGVectorAdapter._remove_tags_from_table.retry_with(wait=wait_none())

    await detag(
        adapter, "Entity_name", text("select"), text("update"), text("delete"), {"tags": ["a"]}
    )

    # The retry starts over from the target selection: select, update, delete.
    assert [str(statement) for statement in executed] == ["select", "update", "delete"]
