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

``create_data_points`` first classifies the rows outside any transaction, to embed only
the ones an upsert would write (``_rows_to_write``), then writes in phases (see
``_write_data_point_rows``): a lock-free classify, new rows through ``ON CONFLICT DO
NOTHING`` (``fetchmany``), then the rows that would change through the ``DO UPDATE``
upsert (``executemany``). The fake connection
answers the classify from a given set of ids it reports as existing and changed.
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
    """Records every write: ``(kind, statement, parameter rows)``, in order. The classify
    reports an id in ``changed`` as existing and changed, any other id as missing; every
    missing row is inserted."""

    def __init__(self, log, changed=frozenset()):
        self.log = log
        self.changed = changed

    def transaction(self):
        return self

    async def start(self):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def fetch(self, statement, ids, payloads):
        self.log.append(("classify", statement, list(ids)))
        return [{"id": i, "missing": i not in self.changed} for i in ids]

    async def fetchmany(self, statement, params):
        self.log.append(("insert", statement, list(params)))
        return [{"id": row[0]} for row in params]

    async def executemany(self, statement, params):
        self.log.append(("upsert", statement, list(params)))


class _Pool:
    def __init__(self, log, changed=frozenset()):
        self.connection = _Connection(log, changed)

    async def acquire(self):
        return self.connection

    async def release(self, connection):
        assert connection is self.connection


def _adapter(log, schema="", changed=frozenset()):
    adapter = object.__new__(PGVectorAdapter)
    adapter.embedding_engine = _Embeddings()
    adapter.schema = schema
    adapter._write_locks = {}
    adapter._write_locks_guard = threading.Lock()

    async def has_collection(name):
        return True

    pool = _Pool(log, changed)

    async def get_write_pool():
        return pool

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


def _writes(log, kind):
    return [(statement, rows) for logged, statement, rows in log if logged == kind]


def _written_ids(log, kind):
    (written,) = _writes(log, kind)
    _statement, rows = written
    return [str(row[0]) for row in rows]


@pytest.mark.asyncio
async def test_create_data_points_writes_rows_in_row_order():
    log = []
    ids = _ids_in_reverse_order(5)
    points = [_Point(id=point_id, text=str(point_id)) for point_id in ids]

    await _adapter(log).create_data_points("Point_text", points)

    assert _written_ids(log, "insert") == sorted(str(i) for i in ids)
    assert _writes(log, "upsert") == []


@pytest.mark.asyncio
async def test_create_data_points_inserts_new_rows_before_upserting_changed_ones():
    """New rows go first through ``DO NOTHING``, then only the changed rows through ``DO
    UPDATE``, each in row order; the classify sees every row once."""
    log = []
    ids = _ids_in_reverse_order(6)
    changed = frozenset(ids[::2])
    points = [_Point(id=point_id, text=str(point_id)) for point_id in ids]

    await _adapter(log, changed=changed).create_data_points("Point_text", points)

    assert [kind for kind, _statement, _rows in log] == [
        "classify",  # before embedding
        "classify",
        "insert",
        "upsert",
    ]
    insert_statement, _rows = _writes(log, "insert")[0]
    assert insert_statement.endswith("ON CONFLICT (id) DO NOTHING RETURNING id")
    upsert_statement, _rows = _writes(log, "upsert")[0]
    assert "ON CONFLICT (id) DO UPDATE SET" in upsert_statement
    assert _written_ids(log, "insert") == sorted(str(i) for i in ids if i not in changed)
    assert _written_ids(log, "upsert") == sorted(str(i) for i in changed)


@pytest.mark.asyncio
async def test_create_data_points_sends_plain_rows_with_numeric_vectors():
    """No ORM object and no text literal: the id, the payload as JSON text, and the
    vector as the numbers the codec encodes in binary."""
    log = []
    point = _Point(text="hello", belongs_to_set=["A"])

    await _adapter(log).create_data_points("Point_text", [point])

    statement, rows = _writes(log, "insert")[0]
    assert statement.startswith(
        'INSERT INTO "Point_text" (id, payload, vector) VALUES ($1, $2, $3)'
    )
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

    _statement, rows = _writes(log, "insert")[0]
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

    assert _written_ids(log, "upsert") == sorted(str(i) for i in ids)
    statement, _rows = _writes(log, "upsert")[0]
    assert "vector = EXCLUDED.vector" in statement


@pytest.mark.asyncio
async def test_a_schema_pinned_adapter_names_its_own_schema():
    """The table is schema-qualified, so a same-named collection in ``public`` is never
    written; the conflict clause refers to the target by its bare name, as Postgres
    requires."""
    log = []
    new, changed = _Point(text="x"), _Point(text="y")

    await _adapter(log, schema="ds_alpha", changed={changed.id}).create_data_points(
        "Point_text", [new, changed]
    )

    (classify, _ids), (classify_again, _ids), (insert, _rows), (upsert, _rows) = [
        (s, r) for _k, s, r in log
    ]
    assert classify_again == classify
    assert 'LEFT JOIN "ds_alpha"."Point_text" ON "Point_text".id = incoming.id' in classify
    assert insert.startswith('INSERT INTO "ds_alpha"."Point_text" ')
    assert upsert.startswith('INSERT INTO "ds_alpha"."Point_text" ')
    for statement in (classify, upsert):
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


@pytest.mark.asyncio
async def test_cancelled_data_point_write_rolls_back_and_releases_pool_and_gate():
    import asyncio

    calls = []

    class Connection(_Connection):
        async def fetch(self, statement, ids, payloads):
            raise asyncio.CancelledError()

        async def rollback(self):
            calls.append("rollback")

    class Pool(_Pool):
        async def release(self, connection):
            calls.append("release")

    pool = Pool([])
    pool.connection = Connection([])
    adapter = _adapter([])

    async def get_pool():
        return pool

    adapter._get_write_pool = get_pool
    gate = asyncio.Semaphore(1)
    with pytest.raises(asyncio.CancelledError):
        await adapter._write_data_point_rows(
            "test", [{"id": uuid4(), "payload": {}, "vector": [1, 2, 3]}], write_gate=gate
        )
    assert calls == ["rollback", "release"]
    assert not gate.locked()
