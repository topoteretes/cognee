"""Re-embedding a pgvector store's collections against a live Postgres.

The store-level steps (``stage_collections``, ``replace_collections``) on the
adapter that needs the most from them: collections share a schema with tables that
are not collections, a collection can be hash-partitioned, each has an HNSW index,
and a vector column's width is part of its type.

No LLM or embedding calls. Skipped unless ``TEST_PGVECTOR_URL`` is set, e.g.:

    TEST_PGVECTOR_URL="postgresql+asyncpg://cognee:cognee@127.0.0.1:5432/cognee_db"
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text

from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage.reembed_collections import (
    STAGING_PREFIX,
    get_reembeddable_collection_names,
    replace_collections,
    stage_collections,
)

PG_URL = os.getenv("TEST_PGVECTOR_URL")

pytestmark = [
    pytest.mark.skipif(not PG_URL, reason="TEST_PGVECTOR_URL not set"),
    pytest.mark.asyncio,
]

try:
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
    )
except ModuleNotFoundError:  # postgres extra not installed
    pytestmark.append(pytest.mark.skip(reason="pgvector extra not installed"))

OLD_WIDTH = 3
NEW_WIDTH = 5


class _Model:
    """An embedding model: axis 0 holds its width, another axis depends on the text."""

    def __init__(self, width: int):
        self.width = width

    def get_vector_size(self) -> int:
        return self.width

    def get_batch_size(self) -> int:
        return 100

    def vector(self, text_: str) -> list[float]:
        values = [float(self.width)] + [0.0] * (self.width - 1)
        values[1 + len(text_) % (self.width - 1)] = 1.0
        return values

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(text_) for text_ in texts]


class _Chunk(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


CHUNKS = [
    _Chunk(text="a", belongs_to_set=["alpha"]),
    _Chunk(text="bb", belongs_to_set=["alpha", "beta"]),
    _Chunk(text="ccc"),
    _Chunk(text="dddd"),
]


@pytest_asyncio.fixture
async def open_store():
    """Open one dataset schema with a given embedding model, as a new engine does."""
    schema = f"ds_reembed_{uuid4().hex[:10]}"
    opened = []

    def _open(model: _Model, partitions: int = 1) -> PGVectorAdapter:
        adapter = PGVectorAdapter(
            connection_string=PG_URL, api_key=None, embedding_engine=model, schema=schema
        )
        adapter.collection_partitions = partitions
        opened.append(adapter)
        return adapter

    first = _open(_Model(OLD_WIDTH))
    async with first.engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    _open.schema = schema
    yield _open
    async with first.engine.begin() as connection:
        await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
    for adapter in opened:
        await adapter.close()


async def _rows(store, collection_name: str) -> dict[str, dict]:
    row_ids = await store.get_data_point_ids(collection_name)
    rows = await store.retrieve(collection_name, row_ids, include_vector=True)
    return {str(row.id): row.payload for row in rows}


async def _vector_types(store, schema: str) -> dict[str, str]:
    """The declared type of every table's ``vector`` column in the schema."""
    async with store.engine.connect() as connection:
        rows = await connection.execute(
            text(
                "SELECT c.relname, format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = :schema AND a.attname = 'vector' AND c.relkind IN ('r', 'p')"
            ),
            {"schema": schema},
        )
        return dict(rows.all())


@pytest.mark.parametrize("partitions", [1, 3], ids=["plain", "partitioned"])
async def test_collections_are_rebuilt_at_the_new_width(open_store, partitions):
    schema = open_store.schema
    old_store = open_store(_Model(OLD_WIDTH), partitions)
    await old_store.index_data_points("_Chunk", "text", CHUNKS)
    await old_store.create_vector_index("_Summary", "text")
    # Tables that are not collections: one without a vector column, one whose
    # ``vector`` column is not a pgvector type.
    async with old_store.engine.begin() as connection:
        await connection.execute(text(f'CREATE TABLE "{schema}".accounts (id int, name text)'))
        await connection.execute(text(f'CREATE TABLE "{schema}".shapes (id int, vector text)'))
    before = await _rows(old_store, "_Chunk_text")

    new = _Model(NEW_WIDTH)
    store = open_store(new, partitions)
    names = await get_reembeddable_collection_names(store)
    assert names == ["_Chunk_text", "_Summary_text"]
    assert await stage_collections(store, names) == len(CHUNKS)
    await replace_collections(store, names)

    assert await get_reembeddable_collection_names(store) == names
    assert not [n for n in await store.get_collection_names() if n.startswith(STAGING_PREFIX)]
    types = await _vector_types(store, schema)
    assert types["_Chunk_text"] == types["_Summary_text"] == f"halfvec({NEW_WIDTH})"
    assert await store.get_stored_vector_size() == NEW_WIDTH

    after = await _rows(store, "_Chunk_text")
    assert set(after) == {str(chunk.id) for chunk in CHUNKS}
    for row_id, payload in after.items():
        assert payload["vector"] == pytest.approx(new.vector(payload["text"]))
        assert {k: v for k, v in payload.items() if k != "vector"} == {
            k: v for k, v in before[row_id].items() if k != "vector"
        }

    # The rebuilt collection is indexed and searched like one created at this width.
    async with store.engine.connect() as connection:
        indexes = await connection.execute(
            text(
                "SELECT count(*) FROM pg_indexes WHERE schemaname = :schema "
                "AND indexdef ILIKE '%hnsw%' AND tablename LIKE :table"
            ),
            {"schema": schema, "table": "_Chunk_text%"},
        )
        assert indexes.scalar() >= 1
    found = await store.search("_Chunk_text", query_text="bb", limit=1, include_payload=True)
    assert found[0].payload["text"] == "bb"
    await store.index_data_points("_Chunk", "text", [_Chunk(text="eeeeee")])
    await store.index_data_points("_Summary", "text", [_Chunk(text="a summary")])
    assert len(await store.get_data_point_ids("_Chunk_text")) == len(CHUNKS) + 1

    # The other tables of the schema were never treated as collections.
    async with store.engine.connect() as connection:
        tables = await connection.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = :schema"), {"schema": schema}
        )
        assert {"accounts", "shapes"} <= set(tables.scalars().all())
