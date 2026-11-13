"""pgvector HNSW indexes against a live Postgres: created, used, complete, migrated.

The unit tests pin the generated SQL; these run it. They check that ``create_collection``
builds the index, that the planner uses it for the adapter's own search query, that a
search returns ``limit`` rows sorted by score and a ``limit=None`` search every row, and
that the migration indexes a collection created before indexes existed, in both the
dedicated-database and the shared-database (per-dataset schema) layout.

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


async def _fill(adapter: PGVectorAdapter, collection: str, count: int) -> None:
    points = [_Point(text=f"point {i}") for i in range(count)]
    await adapter.create_data_points(collection, points)


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


@pytest.mark.parametrize("dimensions", [8, 2048])  # vector, and the halfvec cast
async def test_collections_get_an_index_the_search_uses(dimensions, cleanup):
    adapter = _adapter(dimensions)
    collection = f"HnswLive_{uuid4().hex[:8]}_text"
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))

    await _fill(adapter, collection, 120)
    assert await _index_names(adapter, collection) == [hnsw_index_name(collection)]

    # Capture the adapter's own search statement and EXPLAIN it with its settings.
    captured = []

    def _capture(_conn, _cursor, statement, parameters, _context, _many):
        if "ORDER BY" in statement and collection in statement:
            captured.append((statement, parameters))

    event.listen(adapter.engine.sync_engine, "before_cursor_execute", _capture)
    try:
        query_vector = adapter.embedding_engine.vector("point 7")
        results = await adapter.search(collection, query_vector=query_vector, limit=100)
    finally:
        event.remove(adapter.engine.sync_engine, "before_cursor_execute", _capture)

    # More rows than hnsw.ef_search's default of 40, sorted by the exact distance, and
    # the stored copy of the query vector first.
    assert len(results) == 100
    scores = [result.score for result in results]
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(0, abs=1e-3)

    statement, parameters = captured[-1]
    async with adapter.engine.begin() as conn:
        await conn.execute(text("SET LOCAL enable_seqscan = off"))
        await conn.execute(text("SET LOCAL hnsw.ef_search = 100"))
        plan = await conn.exec_driver_sql("EXPLAIN " + statement, parameters)
        plan_text = "\n".join(row[0] for row in plan)
    assert f'Index Scan using "{hnsw_index_name(collection)}"' in plan_text, plan_text


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
    # The filter runs after the scan. Up to 2,000 dimensions the index is on the column
    # itself, and at this size Postgres picks it for the exact ORDER BY too: it then saw
    # only hnsw.ef_search rows and returned 0 of the 13 tagged ones. A filtered search
    # stays exact. (Above 2,000 dimensions the index is on a halfvec cast, which the
    # exact ORDER BY does not match.)
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


@pytest.mark.parametrize("shared_schema", [False, True])
async def test_the_migration_indexes_collections_created_without_one(shared_schema, cleanup):
    schema = f"ds_hnsw_{uuid4().hex[:8]}" if shared_schema else ""
    adapter = _adapter(2048, schema=schema)
    collection = f"HnswLegacy_{uuid4().hex[:8]}_text"
    if schema:
        async with adapter.engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        cleanup.append((adapter, f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    else:
        cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))

    # A collection as earlier releases created it: primary key only.
    await _fill(adapter, collection, 30)
    async with adapter.engine.begin() as conn:
        await conn.execute(text(f'DROP INDEX "{hnsw_index_name(collection)}"'))
    assert await _index_names(adapter, collection) == []

    context = SimpleNamespace(vector_engine=adapter)
    await pgvector_hnsw_indexes.migrate(context)
    assert await _index_names(adapter, collection) == [hnsw_index_name(collection)]
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

    def statement(table_name, dimensions, version):
        if table_name == broken:
            return f'CREATE INDEX "{hnsw_index_name(table_name)}" ON "{table_name}" USING nope (vector)'
        return real_statement(table_name, dimensions, version)

    monkeypatch.setattr(adapter_module, "hnsw_index_statement", statement)
    result = await adapter.run_migrations()

    assert broken in result["hnsw_index_failed"]
    assert healthy in result["hnsw_index_checked"]
    assert await _index_names(adapter, broken) == []
    assert await _index_names(adapter, healthy) == [hnsw_index_name(healthy)]
