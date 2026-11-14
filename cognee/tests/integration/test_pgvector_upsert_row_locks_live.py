"""pgvector upserts lock only the rows they change, against a live Postgres.

``INSERT ... ON CONFLICT DO UPDATE ... WHERE`` locks every conflicting row until commit,
even when its ``WHERE`` is false. Many concurrent writers upserting the same ids (the
same entity extracted from many documents by pipelines running at once) then queued
behind each other's row locks although nearly every upsert changed nothing.
``create_data_points`` therefore classifies the rows first with a lock-free ``SELECT``,
inserts the missing ones with ``ON CONFLICT DO NOTHING`` and upserts only the rows that
would change. These tests pin that the table ends up exactly as the single upsert left
it, that an unchanged row is neither rewritten nor waited for, that a changed row still
is, and the races between the classify and the writes.

No LLM or embedding calls: the embedding engine is a stub whose vectors differ on every
call, so a kept vector is told apart from a replaced one. Skipped unless
``TEST_PGVECTOR_URL`` is set, e.g.:

    TEST_PGVECTOR_URL="postgresql+asyncpg://cognee:cognee@127.0.0.1:5432/cognee_db"
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text

from cognee.infrastructure.engine import DataPoint

PG_URL = os.getenv("TEST_PGVECTOR_URL")

pytestmark = [
    pytest.mark.skipif(not PG_URL, reason="TEST_PGVECTOR_URL not set"),
    pytest.mark.asyncio,
]

try:
    from cognee.infrastructure.databases.vector.pgvector import PGVectorAdapter as adapter_module
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter
    from cognee.infrastructure.databases.vector.pgvector.serialize_data import serialize_data
except ModuleNotFoundError:  # postgres extra not installed
    pytestmark.append(pytest.mark.skip(reason="pgvector extra not installed"))

DIMENSIONS = 8
# Long enough for any write that waits for nothing; a write that waits for a lock held
# by an open transaction never finishes inside it.
NO_WAIT_SECONDS = 20


class _Embeddings:
    """Embeddings that differ on every call, even for the same text, so a test can tell
    which write's vector a row holds. ``embedded[text]`` lists them in call order."""

    def __init__(self):
        self.embedded: dict[str, list[list[float]]] = {}

    def get_vector_size(self) -> int:
        return DIMENSIONS

    def get_batch_size(self) -> int:
        return 1000

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(value) for value in texts]

    def _embed(self, value: str) -> list[float]:
        calls = self.embedded.setdefault(value, [])
        rng = random.Random(f"{value}:{len(calls)}")
        calls.append([rng.uniform(-1, 1) for _ in range(DIMENSIONS)])
        return calls[-1]


class _Point(DataPoint):
    text: str
    note: str = ""
    metadata: dict = {"index_fields": ["text"]}


@pytest.fixture
def embeddings():
    return _Embeddings()


@pytest_asyncio.fixture
async def adapter(embeddings):
    adapter = PGVectorAdapter(connection_string=PG_URL, api_key=None, embedding_engine=embeddings)
    yield adapter
    await adapter.close()


@pytest_asyncio.fixture
async def collection(adapter):
    name = f"RowLocks_{uuid4().hex[:10]}_text"
    await adapter.create_collection(name, _Point)
    yield name
    await adapter.delete_collection(name)


async def _rows(adapter: PGVectorAdapter, collection: str) -> dict[UUID, dict]:
    """Every row: payload, vector and row version (``ctid``/``xmin`` change on any rewrite)."""
    async with adapter.engine.connect() as conn:
        result = await conn.execute(
            text(
                f'SELECT id, payload::text, vector::text, ctid::text, xmin::text FROM "{collection}"'
            )
        )
        return {
            row_id: {
                "payload": json.loads(payload),
                "vector": json.loads(vector),
                "version": (ctid, xmin),
            }
            for row_id, payload, vector, ctid, xmin in result.all()
        }


def _assert_vector(stored: list[float], expected: list[float]) -> None:
    assert stored == pytest.approx(expected, abs=1e-6)


async def _wait_until_blocked(adapter: PGVectorAdapter, holder_pid: int) -> None:
    """Return once some session waits for a lock held by ``holder_pid``."""
    await asyncio.wait_for(_poll_until_blocked(adapter, holder_pid), NO_WAIT_SECONDS)


async def _poll_until_blocked(adapter: PGVectorAdapter, holder_pid: int) -> None:
    while True:
        async with adapter.engine.connect() as conn:
            waiting = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE :pid = ANY(pg_blocking_pids(pid))"
                    ),
                    {"pid": holder_pid},
                )
            ).scalar_one()
        if waiting:
            return
        await asyncio.sleep(0.05)


# --- end state: exactly what the single upsert produced -------------------------------


async def test_new_rows_are_inserted(adapter, collection, embeddings):
    points = [_Point(text=f"new {i}", belongs_to_set=["A"]) for i in range(5)]

    await adapter.create_data_points(collection, points)

    rows = await _rows(adapter, collection)
    assert set(rows) == {p.id for p in points}
    for point in points:
        assert rows[point.id]["payload"]["text"] == point.text
        assert rows[point.id]["payload"]["belongs_to_set"] == ["A"]
        _assert_vector(rows[point.id]["vector"], embeddings.embedded[point.text][0])


async def test_identical_reupsert_writes_no_new_row_version(adapter, collection, embeddings):
    """Unicode, quotes, backslashes and a stripped NUL: the classify reads the payload
    from JSON text, the upsert from the json column, and both must see it unchanged.

    Tags are ``[]`` as ``index_data_points`` writes them: a stored JSON ``null`` merges
    to ``[]`` and is rewritten once (unchanged by the classify, which shares that SQL).
    """
    points = [
        _Point(text='quote " and \\ backslash', note="ümlaut ✓ 😀 'single'", belongs_to_set=[]),
        _Point(text="nul\x00stripped", note="tab\tnewline\n", belongs_to_set=[]),
        _Point(text="plain", belongs_to_set=["A", "B"]),
    ]
    await adapter.create_data_points(collection, points)
    before = await _rows(adapter, collection)

    await asyncio.sleep(0.01)  # the re-upsert's timestamps differ
    again = [p.model_copy(update={"updated_at": p.updated_at + 5}) for p in points]
    await adapter.create_data_points(collection, again)

    after = await _rows(adapter, collection)
    assert {i: r["version"] for i, r in after.items()} == {
        i: r["version"] for i, r in before.items()
    }
    assert after == before


@pytest.mark.parametrize(
    ("stored", "incoming", "merged"),
    [
        (["A"], ["B"], ["A", "B"]),
        (["B"], ["A"], ["A", "B"]),
        (None, ["A"], ["A"]),
        (["A"], None, ["A"]),
        (None, None, []),
    ],
)
async def test_belongs_to_set_is_unioned(adapter, collection, stored, incoming, merged):
    point_id = uuid4()
    await adapter.create_data_points(
        collection, [_Point(id=point_id, text="t", belongs_to_set=stored)]
    )

    await adapter.create_data_points(
        collection, [_Point(id=point_id, text="t", belongs_to_set=incoming)]
    )

    rows = await _rows(adapter, collection)
    assert sorted(rows[point_id]["payload"]["belongs_to_set"]) == merged


async def test_stored_payload_without_belongs_to_set_gets_the_incoming_tags(adapter, collection):
    point_id = uuid4()
    await adapter.upsert_raw_vectors(
        collection, [{"id": str(point_id), "vector": [0.0] * DIMENSIONS, "payload": {"text": "t"}}]
    )

    await adapter.create_data_points(
        collection, [_Point(id=point_id, text="t", belongs_to_set=["A"])]
    )

    assert (await _rows(adapter, collection))[point_id]["payload"]["belongs_to_set"] == ["A"]


async def test_changed_text_replaces_the_vector_other_changes_keep_it(
    adapter, collection, embeddings
):
    retexted, renoted = uuid4(), uuid4()
    await adapter.create_data_points(
        collection, [_Point(id=retexted, text="old"), _Point(id=renoted, text="kept")]
    )

    await adapter.create_data_points(
        collection,
        [_Point(id=retexted, text="new"), _Point(id=renoted, text="kept", note="changed")],
    )

    rows = await _rows(adapter, collection)
    assert rows[retexted]["payload"]["text"] == "new"
    _assert_vector(rows[retexted]["vector"], embeddings.embedded["new"][0])
    assert rows[renoted]["payload"]["note"] == "changed"
    _assert_vector(rows[renoted]["vector"], embeddings.embedded["kept"][0])


async def test_duplicate_ids_in_one_call_are_written_once_with_tags_unioned(adapter, collection):
    existing, new = uuid4(), uuid4()
    await adapter.create_data_points(
        collection, [_Point(id=existing, text="e", belongs_to_set=["S"])]
    )

    await adapter.create_data_points(
        collection,
        [
            _Point(id=new, text="n", belongs_to_set=["A"]),
            _Point(id=existing, text="e", belongs_to_set=["A"]),
            _Point(id=new, text="n", belongs_to_set=["B"]),
            _Point(id=existing, text="e", belongs_to_set=["B"]),
        ],
    )

    rows = await _rows(adapter, collection)
    assert sorted(rows[new]["payload"]["belongs_to_set"]) == ["A", "B"]
    assert sorted(rows[existing]["payload"]["belongs_to_set"]) == ["A", "B", "S"]


@pytest.mark.parametrize("batch_size", [1000, 3])
async def test_mixed_call_across_statement_batches(
    adapter, collection, embeddings, monkeypatch, batch_size
):
    """New, unchanged, retagged, retexted and renoted rows in one call; with a batch
    size of 3 every phase spans several statements."""
    monkeypatch.setattr(adapter_module, "QUERY_BATCH_SIZE", batch_size)
    unchanged = [_Point(text=f"same {i}", belongs_to_set=["A"]) for i in range(4)]
    retagged = [_Point(text=f"tag {i}", belongs_to_set=["A"]) for i in range(3)]
    retexted = [_Point(text=f"old {i}") for i in range(3)]
    renoted = [_Point(text=f"note {i}") for i in range(3)]
    await adapter.create_data_points(collection, unchanged + retagged + retexted + renoted)
    before = await _rows(adapter, collection)

    new = [_Point(text=f"new {i}", belongs_to_set=["B"]) for i in range(4)]
    call = (
        new
        + [p.model_copy(update={"updated_at": p.updated_at + 5}) for p in unchanged]
        + [_Point(id=p.id, text=p.text, belongs_to_set=["B"]) for p in retagged]
        + [_Point(id=p.id, text=f"new text {i}") for i, p in enumerate(retexted)]
        + [_Point(id=p.id, text=p.text, note="n") for p in renoted]
    )
    random.Random(0).shuffle(call)
    await adapter.create_data_points(collection, call)

    rows = await _rows(adapter, collection)
    assert len(rows) == len(before) + len(new)
    for p in new:
        assert rows[p.id]["payload"]["belongs_to_set"] == ["B"]
        _assert_vector(rows[p.id]["vector"], embeddings.embedded[p.text][0])
    for p in unchanged:
        assert rows[p.id] == before[p.id]
    for p in retagged:
        assert rows[p.id]["version"] != before[p.id]["version"]
        assert sorted(rows[p.id]["payload"]["belongs_to_set"]) == ["A", "B"]
        assert rows[p.id]["vector"] == before[p.id]["vector"]
    for i, p in enumerate(retexted):
        assert rows[p.id]["payload"]["text"] == f"new text {i}"
        _assert_vector(rows[p.id]["vector"], embeddings.embedded[f"new text {i}"][0])
    for p in renoted:
        assert rows[p.id]["payload"]["note"] == "n"
        assert rows[p.id]["vector"] == before[p.id]["vector"]


# --- row locks ------------------------------------------------------------------------


async def test_unchanged_row_locked_by_another_transaction_is_not_waited_for(adapter, collection):
    """The regression: an open transaction holds a row lock on an existing row. An upsert
    that would not change that row, plus new rows, must not wait for it."""
    held = _Point(text="held", belongs_to_set=["A"])
    await adapter.create_data_points(collection, [held])

    async with adapter.engine.connect() as holder:
        await holder.begin()
        await holder.execute(
            text(f'SELECT 1 FROM "{collection}" WHERE id = :id FOR UPDATE'), {"id": held.id}
        )
        try:
            new = [_Point(text=f"new {i}") for i in range(3)]
            same = held.model_copy(update={"updated_at": held.updated_at + 5})
            await asyncio.wait_for(
                adapter.create_data_points(collection, [same, *new]), NO_WAIT_SECONDS
            )
        finally:
            await holder.rollback()

    assert len(await _rows(adapter, collection)) == 4


async def test_changed_row_still_waits_for_the_lock_and_is_updated(adapter, collection):
    held_id = uuid4()
    await adapter.create_data_points(
        collection, [_Point(id=held_id, text="held", belongs_to_set=["A"])]
    )

    async with adapter.engine.connect() as holder:
        await holder.begin()
        holder_pid = (await holder.execute(text("SELECT pg_backend_pid()"))).scalar_one()
        await holder.execute(
            text(f'SELECT 1 FROM "{collection}" WHERE id = :id FOR UPDATE'), {"id": held_id}
        )
        write = asyncio.create_task(
            adapter.create_data_points(
                collection, [_Point(id=held_id, text="held", belongs_to_set=["B"])]
            )
        )
        await _wait_until_blocked(adapter, holder_pid)
        assert not write.done()
        await holder.commit()

    await asyncio.wait_for(write, NO_WAIT_SECONDS)
    rows = await _rows(adapter, collection)
    assert sorted(rows[held_id]["payload"]["belongs_to_set"]) == ["A", "B"]


# --- races between the classify and the writes ----------------------------------------


async def _delete(adapter: PGVectorAdapter, collection: str, point_id: UUID) -> None:
    async with adapter.engine.begin() as deleter:
        await deleter.execute(text(f'DELETE FROM "{collection}" WHERE id = :id'), {"id": point_id})


async def _insert_uncommitted(holder, collection: str, point: _Point) -> int:
    """Insert ``point`` as the adapter would, in a transaction left open on ``holder``;
    returns ``holder``'s backend pid."""
    await holder.begin()
    await holder.execute(
        text(
            f'INSERT INTO "{collection}" (id, payload, vector) '
            "VALUES (:id, CAST(:payload AS json), CAST(:vector AS vector))"
        ),
        {
            "id": point.id,
            "payload": json.dumps(serialize_data(point.model_dump())),
            "vector": str([0.0] * DIMENSIONS),
        },
    )
    return (await holder.execute(text("SELECT pg_backend_pid()"))).scalar_one()


@pytest.mark.parametrize(
    ("theirs", "mine", "merged"), [(["A"], ["B"], ["A", "B"]), (["A"], ["A"], ["A"])]
)
async def test_insert_that_loses_the_race_is_classified_again(
    adapter, collection, theirs, mine, merged
):
    """Another transaction inserts the id after the classify saw it missing: the
    ``DO NOTHING`` insert waits for it, skips the id, and the id is classified again and
    upserted only if that would change it."""
    point_id = uuid4()
    async with adapter.engine.connect() as holder:
        holder_pid = await _insert_uncommitted(
            holder, collection, _Point(id=point_id, text="t", belongs_to_set=theirs)
        )
        theirs_version = tuple(
            (
                await holder.execute(
                    text(f'SELECT ctid::text, xmin::text FROM "{collection}" WHERE id = :id'),
                    {"id": point_id},
                )
            ).one()
        )
        write = asyncio.create_task(
            adapter.create_data_points(
                collection, [_Point(id=point_id, text="t", belongs_to_set=mine)]
            )
        )
        await _wait_until_blocked(adapter, holder_pid)
        await holder.commit()

    await asyncio.wait_for(write, NO_WAIT_SECONDS)

    row = (await _rows(adapter, collection))[point_id]
    assert sorted(row["payload"]["belongs_to_set"]) == merged
    if merged == theirs:
        assert row["version"] == theirs_version
    else:
        assert row["version"] != theirs_version


async def test_row_deleted_after_the_classify_is_inserted_by_the_upsert(adapter, collection):
    """A row classified as changed is deleted before the upsert runs: the upsert, which
    is the general statement, inserts it. The call is held between the two by an
    uncommitted insert of one of its new ids."""
    changed_id, blocking_id = uuid4(), uuid4()
    await adapter.create_data_points(
        collection, [_Point(id=changed_id, text="t", belongs_to_set=["A"])]
    )

    async with adapter.engine.connect() as holder:
        holder_pid = await _insert_uncommitted(
            holder, collection, _Point(id=blocking_id, text="t", belongs_to_set=["Y"])
        )
        write = asyncio.create_task(
            adapter.create_data_points(
                collection,
                [
                    _Point(id=changed_id, text="t", belongs_to_set=["B"]),
                    _Point(id=blocking_id, text="t", belongs_to_set=["X"]),
                ],
            )
        )
        await _wait_until_blocked(adapter, holder_pid)
        try:
            # The delete must not wait for the call: the classify locked nothing.
            await asyncio.wait_for(_delete(adapter, collection, changed_id), NO_WAIT_SECONDS)
        finally:
            await holder.rollback()

    await asyncio.wait_for(write, NO_WAIT_SECONDS)
    rows = await _rows(adapter, collection)
    assert rows[changed_id]["payload"]["belongs_to_set"] == ["B"]
    assert rows[blocking_id]["payload"]["belongs_to_set"] == ["X"]


async def test_concurrent_calls_inserting_the_same_new_ids(embeddings, collection):
    """Two writers insert the same brand-new ids with different tags at once, five times."""
    adapters = [
        PGVectorAdapter(connection_string=PG_URL, api_key=None, embedding_engine=embeddings)
        for _ in range(2)
    ]
    try:
        for _ in range(5):
            ids = [uuid4() for _ in range(200)]
            await asyncio.gather(
                *[
                    a.create_data_points(
                        collection, [_Point(id=i, text=str(i), belongs_to_set=[tag]) for i in ids]
                    )
                    for a, tag in zip(adapters, ["A", "B"])
                ]
            )
        rows = await _rows(adapters[0], collection)
    finally:
        for a in adapters:
            await a.close()

    assert len(rows) == 5 * 200
    assert all(sorted(r["payload"]["belongs_to_set"]) == ["A", "B"] for r in rows.values())
