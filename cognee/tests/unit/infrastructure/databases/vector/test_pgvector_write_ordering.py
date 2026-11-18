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

``create_data_points`` first classifies its rows on a connection of its own, to embed
only the ones an upsert would write, then writes in two phases (see
``_write_data_point_rows``): new rows with ``ON CONFLICT DO NOTHING`` (``fetchmany``),
then the rows that would change with ``ON CONFLICT DO UPDATE`` (``executemany``). The
fake connection answers the lock-free classify query (``fetch``) from given sets of
existing ids.
"""

from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

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
    """Records every write as ``(kind, statement, parameter rows)``, in order.

    The classify query reports an id in ``changed`` as an existing row the upsert would
    change, leaves out an id in ``unchanged``, and reports any other id as missing;
    every missing row is inserted.
    """

    def __init__(self, log, changed=frozenset(), unchanged=frozenset(), events=None):
        self.log = log
        self.changed = changed
        self.unchanged = unchanged
        self.events = events if events is not None else []

    @asynccontextmanager
    async def transaction(self):
        self.events.append("begin")
        yield
        self.events.append("commit")

    async def fetch(self, statement, ids, payloads):
        self.events.append("classify")
        self.classify_statement = statement
        return [{"id": i, "missing": i not in self.changed} for i in ids if i not in self.unchanged]

    async def fetchmany(self, statement, params):
        params = list(params)
        self.log.append(("insert", statement, params))
        return [{"id": row[0]} for row in params]

    async def executemany(self, statement, params):
        self.log.append(("upsert", statement, list(params)))


class _Pool:
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        self.connection.events.append("acquire")
        yield self.connection
        self.connection.events.append("release")


def _adapter(log, schema="", changed=frozenset(), unchanged=frozenset(), events=None):
    adapter = object.__new__(PGVectorAdapter)
    adapter.embedding_engine = _Embeddings()
    adapter.schema = schema
    adapter._write_locks = {}
    adapter._write_locks_guard = threading.Lock()
    adapter.connection = _Connection(log, changed, unchanged, events)

    async def has_collection(name):
        return True

    async def get_write_pool():
        return _Pool(adapter.connection)

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


def _written_ids(entry):
    _kind, _statement, rows = entry
    return [str(row[0]) for row in rows]


@pytest.mark.asyncio
async def test_create_data_points_writes_rows_in_row_order():
    log = []
    ids = _ids_in_reverse_order(5)
    points = [_Point(id=point_id, text=str(point_id)) for point_id in ids]

    await _adapter(log).create_data_points("Point_text", points)

    assert [kind for kind, _, _ in log] == ["insert"]
    assert _written_ids(log[0]) == sorted(str(i) for i in ids)


@pytest.mark.asyncio
async def test_create_data_points_inserts_new_rows_before_upserting_changed_ones():
    """All new rows go first through ``DO NOTHING``, then only the changed rows through
    ``DO UPDATE``, each statement in row order; nothing else is written."""
    log = []
    ids = _ids_in_reverse_order(6)
    changed = frozenset(ids[::2])
    points = [_Point(id=point_id, text=str(point_id)) for point_id in ids]

    await _adapter(log, changed=changed).create_data_points("Point_text", points)

    assert [kind for kind, _, _ in log] == ["insert", "upsert"]
    assert "ON CONFLICT (id) DO NOTHING RETURNING id" in log[0][1]
    assert "ON CONFLICT (id) DO UPDATE SET" in log[1][1]
    assert _written_ids(log[0]) == sorted(str(i) for i in ids if i not in changed)
    assert _written_ids(log[1]) == sorted(str(i) for i in changed)


@pytest.mark.asyncio
async def test_create_data_points_sends_plain_rows_with_numeric_vectors():
    """No ORM object and no text literal: the id, the payload as JSON text, and the
    vector as the numbers the codec encodes in binary."""
    log = []
    new = _Point(text="hello", belongs_to_set=["A"])
    changed = _Point(text="again")

    await _adapter(log, changed={changed.id}).create_data_points("Point_text", [new, changed])

    (_, insert, inserted), (_, upsert, upserted) = log
    for statement in (insert, upsert):
        assert statement.startswith(
            'INSERT INTO "Point_text" (id, payload, vector) VALUES ($1, $2, $3)'
        )
    (row_id, payload, vector) = inserted[0]
    assert row_id == new.id
    assert json.loads(payload)["text"] == "hello"
    assert json.loads(payload)["belongs_to_set"] == ["A"]
    assert vector == [0.1, 0.2, 0.3]
    assert upserted[0][0] == changed.id
    assert upserted[0][2] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_a_short_embedding_response_writes_nothing():
    """Fewer vectors than inputs is an error, not a truncated batch stored as a success."""
    log = []
    adapter = _adapter(log)

    async def one_vector_short(texts):
        return [[0.1, 0.2, 0.3] for _ in texts[:-1]]

    adapter.embed_data = one_vector_short

    with pytest.raises(ValueError, match="returned 1 vectors for 2 inputs"):
        await adapter.create_data_points("Point_text", [_Point(text="a"), _Point(text="b")])

    assert log == []


@pytest.mark.asyncio
async def test_duplicate_ids_in_a_batch_are_written_once_with_their_tags_unioned():
    log = []
    point_id = uuid4()
    points = [
        _Point(id=point_id, text="same", belongs_to_set=["A"]),
        _Point(id=point_id, text="same", belongs_to_set=["B"]),
    ]

    await _adapter(log).create_data_points("Point_text", points)

    _kind, _statement, rows = log[0]
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

    assert [kind for kind, _, _ in log] == ["upsert"]
    assert _written_ids(log[0]) == sorted(str(i) for i in ids)
    assert "vector = EXCLUDED.vector" in log[0][1]


@pytest.mark.asyncio
async def test_a_schema_pinned_adapter_names_its_own_schema():
    """The table is schema-qualified, so a same-named collection in ``public`` is never
    read or written; the conditions refer to the target by its bare name, as Postgres
    requires in the conflict clause."""
    log = []
    new, changed = _Point(text="x"), _Point(text="y")
    adapter = _adapter(log, schema="ds_alpha", changed={changed.id})

    await adapter.create_data_points("Point_text", [new, changed])

    classify = adapter.connection.classify_statement
    (_, insert, _), (_, upsert, _) = log
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
async def test_only_rows_to_write_are_embedded_after_a_read_of_their_own():
    """The rows an upsert would write are found on a connection that is given back,
    with no transaction, before the embedding request; a row stored unchanged is not
    embedded, a duplicate id once."""
    events = []
    ids = _ids_in_reverse_order(3)
    unchanged = ids[0]
    adapter = _adapter([], unchanged=frozenset({unchanged}), events=events)

    async def embed(texts):
        events.append(("embed", sorted(texts)))
        return [[0.1, 0.2, 0.3] for _ in texts]

    adapter.embed_data = embed
    points = [_Point(id=point_id, text=str(point_id)) for point_id in ids]
    duplicate = _Point(id=ids[1], text="last text wins")

    await adapter.create_data_points("Point_text", [*points, duplicate])

    assert events[:4] == [
        "acquire",
        "classify",
        "release",
        ("embed", sorted(["last text wins", str(ids[2])])),
    ]
    assert events[4:7] == ["acquire", "begin", "classify"]


@pytest.mark.asyncio
async def test_nothing_to_write_embeds_nothing():
    log = []
    point = _Point(text="stored")
    adapter = _adapter(log, unchanged=frozenset({point.id}))

    async def embed(texts):
        raise AssertionError("a row stored unchanged must not be embedded")

    adapter.embed_data = embed

    await adapter.create_data_points("Point_text", [point])

    assert log == []


@pytest.mark.asyncio
async def test_a_row_that_needs_a_vector_only_at_write_time_is_embedded_then_written():
    """A row stored unchanged at the first classify and missing inside the write's
    transaction (deleted during the embedding request) has no vector: that transaction
    writes nothing, the row is embedded, and the write runs again with every row."""
    log = []
    events = []
    kept, deleted = _Point(text="kept"), _Point(text="deleted meanwhile")
    adapter = _adapter(log, unchanged=frozenset({deleted.id}), events=events)

    async def embed(texts):
        events.append(("embed", sorted(texts)))
        adapter.connection.unchanged = frozenset()
        return [[0.1, 0.2, 0.3] for _ in texts]

    adapter.embed_data = embed

    await adapter.create_data_points("Point_text", [kept, deleted])

    assert [event for event in events if isinstance(event, tuple)] == [
        ("embed", ["kept"]),
        ("embed", ["deleted meanwhile"]),
    ]
    assert [kind for kind, _, _ in log] == ["insert"]
    assert _written_ids(log[0]) == sorted([str(kept.id), str(deleted.id)])
