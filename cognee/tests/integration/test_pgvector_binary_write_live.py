"""pgvector's binary write path against a live Postgres: same vectors, own schema.

``create_data_points`` and ``upsert_raw_vectors`` send rows over an asyncpg pool with
pgvector's binary codec instead of SQLAlchemy statements with text vector literals.
These tests check what that must keep: a stored vector is exactly what a text insert
of the same numbers stores, a row read back carries the embedding it was written with,
duplicate ids in one batch collapse into one row with their tags unioned, and a
schema-pinned adapter never reads or writes a same-named collection in ``public``.

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
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
    )
except ModuleNotFoundError:  # postgres extra not installed
    pytestmark.append(pytest.mark.skip(reason="pgvector extra not installed"))

DIMENSIONS = 8


class _Embeddings:
    """Deterministic embeddings: the same text always maps to the same vector."""

    def get_vector_size(self) -> int:
        return DIMENSIONS

    def get_batch_size(self) -> int:
        return 100

    def vector(self, value: str) -> list[float]:
        rng = random.Random(value)
        return [rng.uniform(-1, 1) for _ in range(DIMENSIONS)]

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(value) for value in texts]


class _Point(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


def _text_literal(vector: list[float]) -> str:
    """The text form pgvector's SQLAlchemy bind processor used to send."""
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


@pytest_asyncio.fixture
async def adapter():
    adapter = PGVectorAdapter(
        connection_string=PG_URL, api_key=None, embedding_engine=_Embeddings()
    )
    yield adapter
    await adapter.close()


@pytest_asyncio.fixture
async def collection(adapter):
    name = f"BinaryWrite_{uuid4().hex[:10]}_text"
    yield name
    await adapter.delete_collection(name)


async def _stored_text(adapter, table: str, point_id) -> str:
    async with adapter.engine.connect() as connection:
        return (
            await connection.execute(
                text(f"SELECT vector::text FROM {table} WHERE id = :id"), {"id": point_id}
            )
        ).scalar_one()


async def test_binary_write_stores_what_a_text_insert_stores(adapter, collection):
    """Bit for bit: the binary codec's float32 and the server's text parsing agree."""
    point = _Point(text="binary")
    await adapter.create_data_points(collection, [point])

    text_id = uuid4()
    async with adapter.engine.begin() as connection:
        await connection.execute(
            text(
                f'INSERT INTO "{collection}" (id, payload, vector) '
                "VALUES (:id, CAST(:payload AS json), CAST(:vector AS vector))"
            ),
            {
                "id": text_id,
                "payload": "{}",
                "vector": _text_literal(_Embeddings().vector("binary")),
            },
        )

    table = f'"{collection}"'
    assert await _stored_text(adapter, table, point.id) == await _stored_text(
        adapter, table, text_id
    )
    (row,) = await adapter.retrieve(collection, [str(point.id)], include_vector=True)
    assert row.payload["vector"] == pytest.approx(_Embeddings().vector("binary"), abs=1e-6)


async def test_raw_vectors_round_trip_through_the_binary_path(adapter, collection):
    vector = _Embeddings().vector("raw")
    point_id = uuid4()
    await adapter.upsert_raw_vectors(
        collection, [{"id": str(point_id), "vector": vector, "payload": {"text": "raw"}}]
    )

    (row,) = await adapter.retrieve(collection, [str(point_id)], include_vector=True)
    assert row.payload["text"] == "raw"
    assert row.payload["vector"] == pytest.approx(vector, abs=1e-6)


async def test_duplicate_ids_in_one_batch_write_one_row_with_tags_unioned(adapter, collection):
    point_id = uuid4()
    await adapter.create_data_points(
        collection,
        [
            _Point(id=point_id, text="shared", belongs_to_set=["A"]),
            _Point(id=point_id, text="shared", belongs_to_set=["B"]),
        ],
    )

    rows = await adapter.retrieve(collection, [str(point_id)])
    assert len(rows) == 1
    assert sorted(rows[0].payload["belongs_to_set"]) == ["A", "B"]


async def test_the_write_pool_is_kept_while_the_adapter_lives_and_closed_with_it(
    adapter, collection
):
    await adapter.create_data_points(collection, [_Point(text="one")])
    pool = adapter._write_pool
    assert pool is not None
    await adapter.create_data_points(collection, [_Point(text="two")])
    assert adapter._write_pool is pool

    await adapter.close()
    assert adapter._write_pool is None


async def test_a_schema_pinned_adapter_never_touches_a_same_named_public_collection():
    """``public`` stays on the pinned adapter's search path for pgvector's types. A
    collection of the same name there used to be found by ``has_collection`` and then
    written and read; now every statement names the pinned schema."""
    schema = f"ds_{uuid4().hex[:8]}"
    name = f"Pinned_{uuid4().hex[:8]}_text"
    unpinned = PGVectorAdapter(
        connection_string=PG_URL, api_key=None, embedding_engine=_Embeddings()
    )
    async with unpinned.engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        # The decoy: same name in public, a different width, one row already in it.
        await connection.execute(
            text(
                f'CREATE TABLE public."{name}" (id uuid PRIMARY KEY, payload json, vector vector(3))'
            )
        )
        await connection.execute(
            text(f"INSERT INTO public.\"{name}\" VALUES (:id, '{{}}', '[1,2,3]')"),
            {"id": uuid4()},
        )
    pinned = PGVectorAdapter(
        connection_string=PG_URL, api_key=None, embedding_engine=_Embeddings(), schema=schema
    )
    try:
        assert not await pinned.has_collection(name)
        assert await pinned.existing_collections([name]) == []

        points = [_Point(text="alpha"), _Point(text="beta")]
        await pinned.create_data_points(name, points)
        await pinned.create_data_points(name, points)  # the conflict path, in the schema too

        assert await pinned.has_collection(name)
        assert await pinned.existing_collections([name]) == [name]
        async with unpinned.engine.connect() as connection:
            in_public = (
                await connection.execute(text(f'SELECT count(*) FROM public."{name}"'))
            ).scalar_one()
            in_schema = (
                await connection.execute(text(f'SELECT count(*) FROM "{schema}"."{name}"'))
            ).scalar_one()
        assert in_public == 1
        assert in_schema == 2

        rows = await pinned.retrieve(name, [str(point.id) for point in points])
        assert {row.payload["text"] for row in rows} == {"alpha", "beta"}
        # A search reads the pinned table: public's 3-wide column would reject the query.
        found = await pinned.search(name, query_text="alpha", limit=2)
        assert {row.id for row in found} == {point.id for point in points}

        await pinned.delete_collection(name)
        assert not await pinned.has_collection(name)
        async with unpinned.engine.connect() as connection:
            assert (
                await connection.execute(text(f'SELECT count(*) FROM public."{name}"'))
            ).scalar_one() == 1
    finally:
        await pinned.close()
        async with unpinned.engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await connection.execute(text(f'DROP TABLE IF EXISTS public."{name}"'))
        await unpinned.close()
