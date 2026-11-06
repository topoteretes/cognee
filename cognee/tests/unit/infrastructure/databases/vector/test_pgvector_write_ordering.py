"""pgvector inserts write their rows in one global order and take no collection lock.

A multi-row upsert locks conflicting rows in VALUES order, so two statements over the
same ids in different orders can deadlock. ``create_data_points`` and
``upsert_raw_vectors`` sort their rows by ``_row_order`` instead of serializing every
insert of a collection behind a lock; the live test
``cognee/tests/integration/test_pgvector_concurrent_writes_live.py`` shows both the
deadlock and its absence against Postgres. These tests pin the order the statements
carry, with a fake session.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from sqlalchemy import JSON, Column, MetaData, String, Table
from sqlalchemy.dialects import postgresql

pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

from pgvector.sqlalchemy import Vector

from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
    PGVectorAdapter,
    _row_order,
)
from cognee.infrastructure.engine import DataPoint


class _Point(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


class _Embeddings:
    def get_vector_size(self):
        return 3

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _Session:
    """Records the ids of every multi-row insert, in the order the statement carries."""

    def __init__(self, log):
        self.log = log

    async def execute(self, statement):
        compiled = statement.compile(dialect=postgresql.dialect())
        ids = [
            value
            for key, value in sorted(
                compiled.params.items(), key=lambda item: int(item[0].rsplit("_m", 1)[1])
            )
            if key.startswith("id_m")
        ]
        self.log.append(ids)

    async def commit(self):
        pass


def _adapter(log):
    adapter = object.__new__(PGVectorAdapter)
    adapter.embedding_engine = _Embeddings()
    adapter.Vector = Vector
    adapter._write_locks = {}
    adapter._write_locks_guard = threading.Lock()

    async def has_collection(name):
        return True

    @asynccontextmanager
    async def get_async_session():
        yield _Session(log)

    def no_lock(name):
        raise AssertionError("an insert must not take the collection lock")

    adapter.has_collection = has_collection
    adapter.get_async_session = get_async_session
    adapter._get_write_lock = no_lock
    adapter.embed_data = adapter.embedding_engine.embed_text
    return adapter


def _ids_in_reverse_order(count):
    ids = sorted((uuid4() for _ in range(count)), key=_row_order)
    return list(reversed(ids))


@pytest.mark.asyncio
async def test_create_data_points_writes_rows_in_row_order():
    log = []
    ids = _ids_in_reverse_order(5)
    points = [_Point(id=point_id, text=str(point_id)) for point_id in ids]

    await _adapter(log).create_data_points("Point_text", points)

    assert len(log) == 1
    assert [str(i) for i in log[0]] == sorted(str(i) for i in ids)


@pytest.mark.asyncio
async def test_upsert_raw_vectors_writes_rows_in_row_order():
    log = []
    ids = _ids_in_reverse_order(5)
    adapter = _adapter(log)
    table = Table(
        "Point_text", MetaData(), Column("id", String), Column("payload", JSON), Column("vector")
    )

    async def get_table(name):
        return table

    adapter.get_table = get_table

    await adapter.upsert_raw_vectors(
        "Point_text",
        [{"id": str(i), "vector": [0.1, 0.2, 0.3], "payload": {"text": "t"}} for i in ids],
    )

    assert len(log) == 1
    assert log[0] == sorted(str(i) for i in ids)


def test_row_order_is_the_uuid_btree_order():
    # Postgres orders uuid columns bytewise, which is the canonical text order.
    ids = [uuid4() for _ in range(200)]
    assert sorted(ids, key=_row_order) == sorted(ids, key=lambda i: UUID(str(i)).bytes)
