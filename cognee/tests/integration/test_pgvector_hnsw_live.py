"""pgvector HNSW indexes against a live Postgres: created, used, complete, migrated.

The unit tests pin the generated SQL; these run it. They check that ``create_collection``
builds the index, that the planner uses it for the adapter's own search query, that a
search returns ``limit`` rows sorted by score and a ``limit=None`` search every row, and
that the migration indexes a collection created before indexes existed, in both the
dedicated-database and the shared-database (per-dataset schema) layout.

A new collection stores ``halfvec``, indexed on the column up to 4,000 dimensions. One
created by an earlier release with a ``vector`` column keeps it and keeps working as it
did: indexed on the column up to 2,000 dimensions and through a halfvec cast above.

No LLM or embedding calls: the embedding engine is a deterministic stub. Skipped unless
``TEST_PGVECTOR_URL`` is set, e.g.:

    TEST_PGVECTOR_URL="postgresql+asyncpg://cognee:cognee@127.0.0.1:5432/cognee_db"
"""

from __future__ import annotations

import os
import random
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import event, text

from cognee.infrastructure.engine import DataPoint

PG_URL = os.getenv("TEST_PGVECTOR_URL")

pytestmark = [
    pytest.mark.skipif(not PG_URL, reason="TEST_PGVECTOR_URL not set"),
    pytest.mark.asyncio,
]

try:
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
        hnsw_index_name,
        hnsw_index_statement,
    )
    from cognee.modules.migrations.versions import pgvector_hnsw_indexes
except ModuleNotFoundError:  # postgres extra not installed
    pytestmark.append(pytest.mark.skip(reason="pgvector extra not installed"))


class _Embeddings:
    """Deterministic embeddings: the same text always maps to the same vector."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def get_vector_size(self) -> int:
        return self.dimensions

    def get_batch_size(self) -> int:
        return 100

    def vector(self, value: str) -> list[float]:
        rng = random.Random(value)
        return [rng.uniform(-1, 1) for _ in range(self.dimensions)]

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(value) for value in texts]


class _Point(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


def _adapter(dimensions: int, schema: str = "") -> PGVectorAdapter:
    return PGVectorAdapter(
        connection_string=PG_URL,
        api_key=None,
        embedding_engine=_Embeddings(dimensions),
        schema=schema,
    )


async def _index_names(adapter: PGVectorAdapter, table: str) -> list[str]:
    async with adapter.engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = current_schema() AND tablename = :t AND indexdef ILIKE '% USING hnsw %'"
            ),
            {"t": table},
        )
        return [row[0] for row in rows]


async def _fill(adapter: PGVectorAdapter, collection: str, count: int) -> list[_Point]:
    points = [_Point(text=f"point {i}") for i in range(count)]
    await adapter.create_data_points(collection, points)
    return points


async def _column_type(adapter: PGVectorAdapter, table: str) -> str:
    """The declared type of the collection's vector column, as the catalog prints it."""
    async with adapter.engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                    "JOIN pg_class c ON c.oid = a.attrelid "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = current_schema() AND c.relname = :t "
                    "AND a.attname = 'vector'"
                ),
                {"t": table},
            )
        ).scalar_one()


async def _index_definition(adapter: PGVectorAdapter, table: str) -> str:
    async with adapter.engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE schemaname = current_schema() AND tablename = :t "
                    "AND indexdef ILIKE '% USING hnsw %'"
                ),
                {"t": table},
            )
        ).scalar_one()


async def _create_vector_collection(
    adapter: PGVectorAdapter, collection: str, dimensions: int, *, indexed: bool
) -> None:
    """A collection as created before columns were stored as ``halfvec``: a ``vector``
    column, with (when ``indexed``) the index created for it then."""
    async with adapter.engine.begin() as conn:
        await conn.execute(
            text(
                f'CREATE TABLE "{collection}" '
                f"(id uuid PRIMARY KEY, payload json, vector vector({dimensions}))"
            )
        )
        if indexed:
            statement = hnsw_index_statement(collection, dimensions, column_type="vector")
            await conn.execute(text(statement))


async def _assert_search_uses_index(adapter: PGVectorAdapter, collection: str, query_vector):
    """Run the adapter's own search, EXPLAIN its statement with its settings, and return
    the results."""
    captured = []

    def _capture(_conn, _cursor, statement, parameters, _context, _many):
        if "ORDER BY" in statement and collection in statement:
            captured.append((statement, parameters))

    event.listen(adapter.engine.sync_engine, "before_cursor_execute", _capture)
    try:
        results = await adapter.search(collection, query_vector=query_vector, limit=100)
    finally:
        event.remove(adapter.engine.sync_engine, "before_cursor_execute", _capture)

    statement, parameters = captured[-1]
    async with adapter.engine.begin() as conn:
        await conn.execute(text("SET LOCAL enable_seqscan = off"))
        await conn.execute(text("SET LOCAL hnsw.ef_search = 100"))
        plan = await conn.exec_driver_sql("EXPLAIN " + statement, parameters)
        plan_text = "\n".join(row[0] for row in plan)
    assert f'Index Scan using "{hnsw_index_name(collection)}"' in plan_text, plan_text
    return results


@pytest_asyncio.fixture
async def cleanup():
    # The dataset database handlers create the extension before any collection.
    setup = _adapter(8)
    async with setup.engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await setup.engine.dispose()
    created: list[tuple[PGVectorAdapter, str]] = []
    yield created
    for adapter, statement in created:
        async with adapter.engine.begin() as conn:
            await conn.execute(text(statement))
        await adapter.engine.dispose()


@pytest.mark.parametrize("dimensions", [8, 1536, 3072])
async def test_collections_get_an_index_the_search_uses(dimensions, cleanup):
    """A new collection stores ``halfvec`` and is indexed on the column."""
    adapter = _adapter(dimensions)
    collection = f"HnswLive_{uuid4().hex[:8]}_text"
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))

    points = await _fill(adapter, collection, 120)
    assert await _index_names(adapter, collection) == [hnsw_index_name(collection)]
    assert await _column_type(adapter, collection) == f"halfvec({dimensions})"
    assert "USING hnsw (vector halfvec_cosine_ops)" in await _index_definition(adapter, collection)

    query_vector = adapter.embedding_engine.vector("point 7")
    results = await _assert_search_uses_index(adapter, collection, query_vector)

    # More rows than hnsw.ef_search's default of 40, sorted by distance, and the stored
    # copy of the query vector first.
    assert len(results) == 100
    scores = [result.score for result in results]
    assert scores == sorted(scores)
    assert results[0].id == points[7].id
    assert scores[0] == pytest.approx(0, abs=1e-3)


async def test_a_collection_too_wide_to_index_is_halfvec_and_searched_exactly(cleanup):
    adapter = _adapter(4097)
    collection = f"HnswWide_{uuid4().hex[:8]}_text"
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))

    points = await _fill(adapter, collection, 60)
    assert await _column_type(adapter, collection) == "halfvec(4097)"
    assert await _index_names(adapter, collection) == []
    assert collection not in (await adapter.run_migrations())["hnsw_index_checked"]

    results = await adapter.search(
        collection, query_vector=adapter.embedding_engine.vector("point 7"), limit=50
    )
    assert len(results) == 50
    assert results[0].id == points[7].id


@pytest.mark.parametrize("dimensions", [8, 3072])
async def test_a_halfvec_collection_reads_scores_and_deletes(dimensions, cleanup):
    """Everything that reads or writes the column works on a halfvec collection: the
    stored width, vectors read back (half precision), scoring by id, an exact search with
    no limit, a node-set filtered search, raw vectors, deletes."""
    # Its own schema, so the stored width is read from this collection alone.
    schema = f"ds_half_{uuid4().hex[:8]}"
    adapter = _adapter(dimensions, schema=schema)
    async with adapter.engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    cleanup.append((adapter, f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    collection = f"HnswHalf_{uuid4().hex[:8]}_text"
    points = [
        _Point(text=f"point {i}", belongs_to_set=["tagged"] if i % 50 == 0 else [])
        for i in range(600)
    ]
    await adapter.create_data_points(collection, points)
    assert await _column_type(adapter, collection) == f"halfvec({dimensions})"
    assert await adapter.get_stored_vector_size() == dimensions

    # Half precision: a value in [-1, 1] is off by < 4.9e-4.
    query_vector = adapter.embedding_engine.vector("point 3")
    (row,) = await adapter.retrieve(collection, [str(points[3].id)], include_vector=True)
    assert all(isinstance(value, float) for value in row.payload["vector"])
    assert row.payload["vector"] == pytest.approx(query_vector, abs=1e-3)

    (score,) = await adapter.score_by_ids(collection, [str(points[3].id)], query_vector)
    assert score.score == pytest.approx(0, abs=1e-3)
    (batch,) = await adapter.score_by_ids_batch(collection, [str(points[3].id)], [query_vector])
    assert batch[0].score == pytest.approx(0, abs=1e-3)

    # The index is on the column, so a search that must be exact turns index scans off:
    # the planner would otherwise answer the exact ORDER BY from the index and return at
    # most hnsw.ef_search rows.
    statements = []

    def _capture(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(adapter.engine.sync_engine, "before_cursor_execute", _capture)
    try:
        everything = await adapter.search(collection, query_vector=query_vector, limit=None)
        tagged = await adapter.search(
            collection, query_vector=query_vector, limit=20, node_name=["tagged"]
        )
    finally:
        event.remove(adapter.engine.sync_engine, "before_cursor_execute", _capture)
    assert len(everything) == 600
    assert everything[0].id == points[3].id
    assert len(tagged) == 12
    assert statements.count("SET LOCAL enable_indexscan = off") == 2

    raw_id = uuid4()
    await adapter.upsert_raw_vectors(
        collection, [{"id": str(raw_id), "vector": [0.5] * dimensions, "payload": {"text": "raw"}}]
    )
    (raw,) = await adapter.retrieve(collection, [str(raw_id)], include_vector=True)
    assert raw.payload["vector"] == [0.5] * dimensions

    await adapter.delete_data_points(collection, [points[3].id, raw_id])
    assert await adapter.retrieve(collection, [str(points[3].id), str(raw_id)]) == []


@pytest.mark.parametrize(
    ("dimensions", "index"),
    [(8, "USING hnsw (vector vector_cosine_ops)"), (3072, "((vector)::halfvec(3072))")],
)
async def test_a_vector_collection_keeps_working(dimensions, index, cleanup):
    """A collection created with a ``vector`` column keeps it, full precision included:
    writes, upserts, the search through its index, scoring, reads, the index rebuild and
    deletes all work as before."""
    adapter = _adapter(dimensions)
    collection = f"HnswVector_{uuid4().hex[:8]}_text"
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))
    await _create_vector_collection(adapter, collection, dimensions, indexed=True)

    points = await _fill(adapter, collection, 120)
    # An upsert that changes a row's text re-embeds it; raw vectors are written as given.
    await adapter.create_data_points(collection, [_Point(id=points[5].id, text="changed")])
    raw_id = uuid4()
    await adapter.upsert_raw_vectors(
        collection, [{"id": str(raw_id), "vector": [0.1] * dimensions, "payload": {"text": "raw"}}]
    )
    assert await _column_type(adapter, collection) == f"vector({dimensions})"
    assert index in await _index_definition(adapter, collection)

    query_vector = adapter.embedding_engine.vector("point 7")
    results = await _assert_search_uses_index(adapter, collection, query_vector)
    assert results[0].id == points[7].id
    assert results[0].score == pytest.approx(0, abs=1e-6)

    changed, raw = await adapter.retrieve(
        collection, [str(points[5].id), str(raw_id)], include_vector=True
    )
    by_id = {str(row.id): row.payload["vector"] for row in (changed, raw)}
    assert by_id[str(points[5].id)] == pytest.approx(
        adapter.embedding_engine.vector("changed"), abs=1e-6
    )
    assert by_id[str(raw_id)] == pytest.approx([0.1] * dimensions, abs=1e-6)
    (score,) = await adapter.score_by_ids(collection, [str(points[7].id)], query_vector)
    assert score.score == pytest.approx(0, abs=1e-6)
    assert len(await adapter.search(collection, query_vector=query_vector, limit=None)) == 121

    async with adapter.engine.begin() as conn:
        await conn.execute(text(f'DROP INDEX "{hnsw_index_name(collection)}"'))
    assert collection in (await adapter.run_migrations())["hnsw_index_checked"]
    assert index in await _index_definition(adapter, collection)
    assert await _column_type(adapter, collection) == f"vector({dimensions})"

    await adapter.delete_data_points(collection, [points[7].id])
    assert await adapter.retrieve(collection, [str(points[7].id)]) == []


async def test_a_search_without_limit_returns_every_row(cleanup):
    adapter = _adapter(8)
    collection = f"HnswLive_{uuid4().hex[:8]}_text"
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))
    await _fill(adapter, collection, 150)

    results = await adapter.search(
        collection, query_vector=adapter.embedding_engine.vector("x"), limit=None
    )
    assert len(results) == 150


async def test_a_node_set_filtered_search_finds_every_tagged_row(cleanup):
    # The filter runs after the scan. The index is on the column itself, and at this size
    # Postgres picks it for the exact ORDER BY too: it then saw only hnsw.ef_search rows
    # and returned 0 of the 13 tagged ones. A filtered search stays exact. (A ``vector``
    # column above 2,000 dimensions is indexed on a halfvec cast, which the exact ORDER
    # BY does not match.)
    adapter = _adapter(8)
    collection = f"HnswLive_{uuid4().hex[:8]}_text"
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))
    points = [
        _Point(text=f"point {i}", belongs_to_set=["tagged"] if i % 400 == 0 else [])
        for i in range(5000)
    ]
    await adapter.create_data_points(collection, points)

    results = await adapter.search(
        collection,
        query_vector=adapter.embedding_engine.vector("x"),
        limit=20,
        node_name=["tagged"],
    )
    assert len(results) == 13


@pytest.mark.parametrize("dimensions", [8, 2048])
@pytest.mark.parametrize("shared_schema", [False, True])
async def test_the_migration_indexes_collections_created_without_one(
    shared_schema, dimensions, cleanup
):
    schema = f"ds_hnsw_{uuid4().hex[:8]}" if shared_schema else ""
    adapter = _adapter(dimensions, schema=schema)
    collection = f"HnswLegacy_{uuid4().hex[:8]}_text"
    if schema:
        async with adapter.engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        cleanup.append((adapter, f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    else:
        cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))

    # A collection as earlier releases created it: a ``vector`` column, primary key only.
    await _create_vector_collection(adapter, collection, dimensions, indexed=False)
    await _fill(adapter, collection, 30)
    assert await _index_names(adapter, collection) == []

    context = SimpleNamespace(vector_engine=adapter)
    await pgvector_hnsw_indexes.migrate(context)
    assert await _index_names(adapter, collection) == [hnsw_index_name(collection)]
    indexed = "vector vector_cosine_ops" if dimensions <= 2000 else "((vector)::halfvec(2048))"
    assert indexed in await _index_definition(adapter, collection)
    assert await _column_type(adapter, collection) == f"vector({dimensions})"
    if schema:
        async with adapter.engine.connect() as conn:
            located = await conn.execute(
                text("SELECT schemaname FROM pg_indexes WHERE indexname = :n"),
                {"n": hnsw_index_name(collection)},
            )
            assert [row[0] for row in located] == [schema]

    await pgvector_hnsw_indexes.downgrade(context)
    assert await _index_names(adapter, collection) == []


@pytest.mark.parametrize("shared_schema", [False, True])
async def test_a_version_change_rebuilds_a_missing_index(shared_schema, cleanup):
    # The migration logs and skips a table whose build fails; the adapter's storage
    # sync, which the runner calls on every cognee version change, retries it.
    from cognee.modules.migrations.versions import adapter_storage_migration

    schema = f"ds_hnsw_{uuid4().hex[:8]}" if shared_schema else ""
    adapter = _adapter(8, schema=schema)
    collection = f"HnswRetry_{uuid4().hex[:8]}_text"
    if schema:
        async with adapter.engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        cleanup.append((adapter, f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    else:
        cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))

    await _fill(adapter, collection, 30)
    async with adapter.engine.begin() as conn:
        await conn.execute(text(f'DROP INDEX "{hnsw_index_name(collection)}"'))

    await adapter_storage_migration.migrate(SimpleNamespace(vector_engine=adapter))
    assert await _index_names(adapter, collection) == [hnsw_index_name(collection)]

    # Idempotent: an indexed store is left as it is.
    result = await adapter.run_migrations()
    assert result["hnsw_index_failed"] == []
    assert await _index_names(adapter, collection) == [hnsw_index_name(collection)]


async def test_a_failed_build_is_logged_and_the_rest_are_indexed(monkeypatch, cleanup):
    # The runner blocks writes to a dataset when run_migrations raises, so one table
    # that cannot be indexed must not stop the others or fail the sync.
    import cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter as adapter_module

    adapter = _adapter(8)
    broken, healthy = (f"HnswFail_{uuid4().hex[:8]}_{name}" for name in ("a", "b"))
    for collection in (broken, healthy):
        cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))
        await _fill(adapter, collection, 10)
        async with adapter.engine.begin() as conn:
            await conn.execute(text(f'DROP INDEX "{hnsw_index_name(collection)}"'))

    real_statement = adapter_module.hnsw_index_statement

    def statement(table_name, dimensions, *, column_type):
        if table_name == broken:
            return f'CREATE INDEX "{hnsw_index_name(table_name)}" ON "{table_name}" USING nope (vector)'
        return real_statement(table_name, dimensions, column_type=column_type)

    monkeypatch.setattr(adapter_module, "hnsw_index_statement", statement)
    result = await adapter.run_migrations()

    assert broken in result["hnsw_index_failed"]
    assert healthy in result["hnsw_index_checked"]
    assert await _index_names(adapter, broken) == []
    assert await _index_names(adapter, healthy) == [hnsw_index_name(healthy)]


@pytest.mark.parametrize("dimensions", [8, 3072])
async def test_a_halfvec_collection_is_left_to_the_adapter_by_the_frozen_migration(
    dimensions, cleanup
):
    """The frozen migration reads ``vector`` columns only, so it passes a halfvec
    collection by, building nothing and dropping nothing; the adapter's index sync builds
    its index, on the column."""
    adapter = _adapter(dimensions)
    collection = f"HnswHalfSync_{uuid4().hex[:8]}_text"
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))
    await _fill(adapter, collection, 30)
    async with adapter.engine.begin() as conn:
        await conn.execute(text(f'DROP INDEX "{hnsw_index_name(collection)}"'))

    context = SimpleNamespace(vector_engine=adapter)
    await pgvector_hnsw_indexes.migrate(context)
    assert await _index_names(adapter, collection) == []

    result = await adapter.run_migrations()
    assert collection in result["hnsw_index_checked"]
    assert "USING hnsw (vector halfvec_cosine_ops)" in await _index_definition(adapter, collection)

    await pgvector_hnsw_indexes.downgrade(context)
    assert await _index_names(adapter, collection) == [hnsw_index_name(collection)]
