"""pgvector writes send plain rows over the binary write pool, in one global order, and
take no collection lock.

A multi-row upsert locks conflicting rows in the order it reaches them, so two
statements over the same ids in different orders can deadlock. ``create_data_points``
and ``upsert_raw_vectors`` sort their rows by ``_row_order`` instead of serializing every
insert of a collection behind a lock; the live test
``cognee/tests/integration/test_pgvector_concurrent_writes_live.py`` shows both the
deadlock and its absence against Postgres. These tests pin what the statements carry,
with a fake write pool: the rows in ``_row_order``, each as ``(id, payload json, vector)``
with the vector left as numbers for pgvector's binary codec, and the table named with
the adapter's schema when it is pinned to one.
"""

from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import numpy as np
import pytest

pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

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


class _Connection:
    """Records every executemany: the statement and its parameter rows, in order."""

    def __init__(self, log):
        self.log = log

    def transaction(self):
        return self

    async def start(self):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def executemany(self, statement, params):
        self.log.append((statement, list(params)))


class _Pool:
    def __init__(self, log):
        self.connection = _Connection(log)

    async def acquire(self):
        return self.connection

    async def release(self, connection):
        assert connection is self.connection


def _adapter(log, schema=""):
    adapter = object.__new__(PGVectorAdapter)
    adapter.embedding_engine = _Embeddings()
    adapter.schema = schema
    adapter._write_locks = {}
    adapter._write_locks_guard = threading.Lock()

    async def has_collection(name):
        return True

    async def get_write_pool():
        return _Pool(log)

    def no_lock(name):
        raise AssertionError("an insert must not take the collection lock")

    adapter.has_collection = has_collection
    adapter._get_write_pool = get_write_pool
    adapter._get_write_lock = no_lock
    adapter.embed_data = adapter.embedding_engine.embed_text
    return adapter


def _ids_in_reverse_order(count):
    ids = sorted((uuid4() for _ in range(count)), key=_row_order)
    return list(reversed(ids))


def _written_ids(log):
    assert len(log) == 1, "one executemany per write"
    _statement, rows = log[0]
    return [str(row[0]) for row in rows]


@pytest.mark.asyncio
async def test_create_data_points_writes_rows_in_row_order():
    log = []
    ids = _ids_in_reverse_order(5)
    points = [_Point(id=point_id, text=str(point_id)) for point_id in ids]

    await _adapter(log).create_data_points("Point_text", points)

    assert _written_ids(log) == sorted(str(i) for i in ids)


@pytest.mark.asyncio
async def test_create_data_points_sends_plain_rows_with_numeric_vectors():
    """No ORM object and no text literal: the id, the payload as JSON text, and the
    vector as the numbers the codec encodes in binary."""
    log = []
    point = _Point(text="hello", belongs_to_set=["A"])

    await _adapter(log).create_data_points("Point_text", [point])

    statement, rows = log[0]
    assert statement.startswith(
        'INSERT INTO "Point_text" (id, payload, vector) VALUES ($1, $2, $3)'
    )
    assert "ON CONFLICT (id) DO UPDATE SET" in statement
    (row_id, payload, vector) = rows[0]
    assert row_id == point.id
    assert json.loads(payload)["text"] == "hello"
    assert json.loads(payload)["belongs_to_set"] == ["A"]
    np.testing.assert_array_equal(vector, np.array([0.1, 0.2, 0.3], dtype=np.float32))


@pytest.mark.asyncio
async def test_duplicate_ids_in_a_batch_are_written_once_with_their_tags_unioned():
    log = []
    point_id = uuid4()
    points = [
        _Point(id=point_id, text="same", belongs_to_set=["A"]),
        _Point(id=point_id, text="same", belongs_to_set=["B"]),
    ]

    await _adapter(log).create_data_points("Point_text", points)

    _statement, rows = log[0]
    assert len(rows) == 1
    assert json.loads(rows[0][1])["belongs_to_set"] == ["A", "B"]


@pytest.mark.asyncio
async def test_upsert_raw_vectors_writes_rows_in_row_order():
    log = []
    ids = _ids_in_reverse_order(5)

    await _adapter(log).upsert_raw_vectors(
        "Point_text",
        [{"id": str(i), "vector": [0.1, 0.2, 0.3], "payload": {"text": "t"}} for i in ids],
    )

    assert _written_ids(log) == sorted(str(i) for i in ids)
    statement, _rows = log[0]
    assert "vector = EXCLUDED.vector" in statement


@pytest.mark.asyncio
async def test_a_schema_pinned_adapter_names_its_own_schema():
    """The table is schema-qualified, so a same-named collection in ``public`` is never
    written; the conflict clause refers to the target by its bare name, as Postgres
    requires."""
    log = []

    await _adapter(log, schema="ds_alpha").create_data_points("Point_text", [_Point(text="x")])

    statement, _rows = log[0]
    assert statement.startswith('INSERT INTO "ds_alpha"."Point_text" ')
    assert '"Point_text".payload' in statement
    assert '"ds_alpha"."Point_text".payload' not in statement


def test_row_order_is_the_uuid_btree_order():
    # Postgres orders uuid columns bytewise, which is the canonical text order.
    ids = [uuid4() for _ in range(200)]
    assert sorted(ids, key=_row_order) == sorted(ids, key=lambda i: UUID(str(i)).bytes)


@pytest.mark.asyncio
async def test_cancelled_write_rolls_back_and_releases_pool_and_gate():
    import asyncio

    adapter = _adapter([])
    calls = []

    class Connection(_Connection):
        async def executemany(self, statement, params):
            raise asyncio.CancelledError()

        async def rollback(self):
            calls.append("rollback")

    class Pool(_Pool):
        async def release(self, connection):
            calls.append("release")

    pool = Pool([])
    pool.connection = Connection([])

    async def get_pool():
        return pool

    adapter._get_write_pool = get_pool
    gate = asyncio.Semaphore(1)
    with pytest.raises(asyncio.CancelledError):
        await adapter._write_rows(
            "test",
            [{"id": uuid4(), "payload": {}, "vector": [1, 2, 3]}],
            adapter._data_point_upsert_sql,
            write_gate=gate,
        )
    assert calls == ["rollback", "release"]
    assert not gate.locked()
