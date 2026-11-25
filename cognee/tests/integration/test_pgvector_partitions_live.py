"""Hash-partitioned pgvector collections against a live Postgres.

With ``PGVECTOR_COLLECTION_PARTITIONS`` above 1 a new collection's table is created
``PARTITION BY HASH (id)``. These tests check what that has to keep true: every
partition has its own valid HNSW index, rows are written, found, replaced and removed
as in a plain table, the collection is one collection wherever collections are listed,
dropping it drops its partitions, and with the default of 1 the table is the plain one
it always was.

No LLM or embedding calls: the embedding engine is a deterministic stub. Skipped unless
``TEST_PGVECTOR_URL`` is set, e.g.:

    TEST_PGVECTOR_URL="postgresql+asyncpg://cognee:cognee@127.0.0.1:5432/cognee_db"
"""

from __future__ import annotations

import os
import random
from uuid import uuid4

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
    from cognee.infrastructure.databases.vector.config import get_vectordb_config
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
        hnsw_index_name,
        partition_name,
    )
except ModuleNotFoundError:  # postgres extra not installed
    pytestmark.append(pytest.mark.skip(reason="pgvector extra not installed"))

PARTITIONS = 4


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


def _adapter(partitions: int = PARTITIONS, schema: str = "") -> PGVectorAdapter:
    adapter = PGVectorAdapter(
        connection_string=PG_URL,
        api_key=None,
        embedding_engine=_Embeddings(8),
        schema=schema,
    )
    adapter.collection_partitions = partitions
    return adapter


def _collection() -> str:
    return f"PartitionLive_{uuid4().hex[:8]}_text"


async def _relkind(adapter: PGVectorAdapter, table: str) -> str | None:
    """'p' for a partitioned table, 'r' for a plain one, None when there is none."""
    async with adapter.engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT c.relkind::text FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = current_schema() AND c.relname = :t"
                ),
                {"t": table},
            )
        ).scalar()


async def _partitions(adapter: PGVectorAdapter, table: str) -> dict[str, tuple[int, bool]]:
    """Per partition of ``table``: (its rows, whether it has a valid HNSW index)."""
    async with adapter.engine.connect() as conn:
        names = (
            await conn.execute(
                text(
                    "SELECT child.relname FROM pg_inherits i "
                    "JOIN pg_class parent ON parent.oid = i.inhparent "
                    "JOIN pg_class child ON child.oid = i.inhrelid "
                    "JOIN pg_namespace n ON n.oid = parent.relnamespace "
                    "WHERE n.nspname = current_schema() AND parent.relname = :t "
                    "ORDER BY child.relname"
                ),
                {"t": table},
            )
        ).scalars()
        found = {}
        for name in list(names):
            rows = (await conn.execute(text(f'SELECT count(*) FROM ONLY "{name}"'))).scalar()
            indexed = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM pg_index x "
                        "JOIN pg_class index ON index.oid = x.indexrelid "
                        "JOIN pg_class owner ON owner.oid = x.indrelid "
                        "JOIN pg_am am ON am.oid = index.relam "
                        "JOIN pg_namespace n ON n.oid = owner.relnamespace "
                        "WHERE n.nspname = current_schema() AND owner.relname = :t "
                        "AND am.amname = 'hnsw' AND x.indisvalid AND x.indisready"
                    ),
                    {"t": name},
                )
            ).scalar()
            found[name] = (rows, indexed == 1)
        return found


@pytest_asyncio.fixture
async def cleanup():
    # The dataset database handlers create the extension before any collection.
    setup = _adapter()
    async with setup.engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await setup.engine.dispose()
    created: list[tuple[PGVectorAdapter, str]] = []
    yield created
    for adapter, statement in created:
        async with adapter.engine.begin() as conn:
            await conn.execute(text(statement))
        await adapter.engine.dispose()


async def test_a_partitioned_collection_has_an_index_on_every_partition(cleanup):
    adapter, collection = _adapter(), _collection()
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))

    await adapter.create_collection(collection)

    assert await _relkind(adapter, collection) == "p"
    assert await _partitions(adapter, collection) == {
        partition_name(collection, remainder): (0, True) for remainder in range(PARTITIONS)
    }
    # The parent carries the collection's index name; the partitions' are attached to it.
    async with adapter.engine.connect() as conn:
        assert (
            await conn.execute(
                text(
                    "SELECT count(*) FROM pg_inherits i "
                    "JOIN pg_class parent ON parent.oid = i.inhparent "
                    "WHERE parent.relname = :index"
                ),
                {"index": hnsw_index_name(collection)},
            )
        ).scalar() == PARTITIONS


async def test_rows_of_a_partitioned_collection_are_written_found_replaced_and_removed(cleanup):
    adapter, collection = _adapter(), _collection()
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))
    points = [_Point(text=f"point {i}") for i in range(200)]

    await adapter.create_data_points(collection, points)

    # Every row is stored once, spread over the partitions.
    stored = await _partitions(adapter, collection)
    assert sum(rows for rows, _ in stored.values()) == 200
    assert all(rows > 0 for rows, _ in stored.values())

    # The nearest row to a stored text is that row, whichever partition holds it.
    results = await adapter.search(
        collection, query_vector=adapter.embedding_engine.vector("point 7"), limit=10
    )
    assert str(results[0].id) == str(points[7].id)
    assert results[0].score == pytest.approx(0, abs=1e-3)
    assert [result.score for result in results] == sorted(result.score for result in results)
    everything = await adapter.search(
        collection, query_vector=adapter.embedding_engine.vector("x"), limit=None
    )
    assert len(everything) == 200

    # Rows are read back by id from any partition.
    wanted = [points[3], points[150]]
    retrieved = await adapter.retrieve(collection, [str(point.id) for point in wanted])
    assert sorted(str(row.id) for row in retrieved) == sorted(str(point.id) for point in wanted)

    # Writing the same ids again replaces them: the primary key still decides.
    await adapter.create_data_points(collection, points[:50])
    assert sum(rows for rows, _ in (await _partitions(adapter, collection)).values()) == 200

    await adapter.delete_data_points(collection, [point.id for point in points[:120]])
    assert sum(rows for rows, _ in (await _partitions(adapter, collection)).values()) == 80


@pytest.mark.parametrize("shared_schema", [False, True])
async def test_a_partitioned_collection_is_one_collection_wherever_they_are_listed(
    shared_schema, cleanup
):
    schema = f"ds_part_{uuid4().hex[:8]}" if shared_schema else ""
    adapter, collection = _adapter(schema=schema), _collection()
    if schema:
        async with adapter.engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        cleanup.append((adapter, f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    else:
        cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))
    await adapter.create_data_points(collection, [_Point(text=f"point {i}") for i in range(40)])

    # Listed once, as the collection: the partitions are not tables of their own.
    listed = [name.rsplit(".", 1)[-1] for name in await adapter.get_table_names()]
    assert listed.count(collection) == 1
    assert not [name for name in listed if name.startswith(f"{collection}_p")]
    assert await adapter.has_collection(collection)

    # The index check on a version change sees the collection, not its partitions.
    checked = await adapter.run_migrations()
    assert collection in checked["hnsw_index_checked"]
    assert not [
        name for name in checked["hnsw_index_checked"] if name.startswith(f"{collection}_p")
    ]
    assert checked["hnsw_index_failed"] == []
    assert all(indexed for _, indexed in (await _partitions(adapter, collection)).values())


async def test_dropping_a_partitioned_collection_drops_its_partitions(cleanup):
    adapter, collection = _adapter(), _collection()
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))
    await adapter.create_data_points(collection, [_Point(text=f"point {i}") for i in range(40)])

    await adapter.delete_collection(collection)

    assert await _relkind(adapter, collection) is None
    for remainder in range(PARTITIONS):
        assert await _relkind(adapter, partition_name(collection, remainder)) is None
    assert not await adapter.has_collection(collection)


async def test_one_partition_is_the_plain_table_it_always_was(cleanup):
    adapter, collection = _adapter(partitions=1), _collection()
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))

    await adapter.create_data_points(collection, [_Point(text=f"point {i}") for i in range(20)])

    assert await _relkind(adapter, collection) == "r"
    assert await _partitions(adapter, collection) == {}


async def test_the_setting_decides_how_a_new_collection_is_created(monkeypatch, cleanup):
    monkeypatch.setenv("PGVECTOR_COLLECTION_PARTITIONS", "3")
    get_vectordb_config.cache_clear()
    try:
        adapter = PGVectorAdapter(
            connection_string=PG_URL, api_key=None, embedding_engine=_Embeddings(8)
        )
    finally:
        monkeypatch.delenv("PGVECTOR_COLLECTION_PARTITIONS")
        get_vectordb_config.cache_clear()
    collection = _collection()
    cleanup.append((adapter, f'DROP TABLE IF EXISTS "{collection}"'))

    await adapter.create_collection(collection)

    assert adapter.collection_partitions == 3
    assert len(await _partitions(adapter, collection)) == 3
