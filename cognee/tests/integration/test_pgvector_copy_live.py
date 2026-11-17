"""COPY has the binary writer's values, conflict semantics and atomicity."""

import asyncio
import os
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text

from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter
from cognee.infrastructure.engine import DataPoint

URL = os.getenv("TEST_PGVECTOR_URL")
pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(not URL, reason="TEST_PGVECTOR_URL unset")]


class Point(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


class Embeddings:
    def get_vector_size(self):
        return 8

    async def embed_text(self, texts):
        return [[0.123456789] * (7 if value == "bad" else 8) for value in texts]


@pytest_asyncio.fixture
async def adapter():
    value = PGVectorAdapter(URL, None, Embeddings())
    yield value
    await value.close()


@pytest_asyncio.fixture
async def names(adapter):
    values = ["Copy_" + uuid4().hex[:12] for _ in range(2)]
    yield values
    for name in values:
        if await adapter.has_collection(name):
            await adapter.delete_collection(name)


async def rows(adapter, name):
    async with adapter.engine.connect() as connection:
        return (
            await connection.execute(
                text(f'SELECT id, payload::jsonb, vector_send(vector) FROM "{name}" ORDER BY id')
            )
        ).all()


async def test_copy_is_identical_to_binary_insert_and_reingestion(adapter, names):
    points = [Point(text=str(i), belongs_to_set=["first"]) for i in range(128)]
    for use_copy, name in zip((False, True), names):
        await adapter.create_data_points(name, points, use_copy=use_copy)
    assert await rows(adapter, names[0]) == await rows(adapter, names[1])
    updates = [p.model_copy(update={"belongs_to_set": ["second"]}) for p in points]
    for use_copy, name in zip((False, True), names):
        await adapter.create_data_points(name, updates, use_copy=use_copy)
    assert await rows(adapter, names[0]) == await rows(adapter, names[1])
    assert set(
        (await adapter.retrieve(names[1], [str(points[0].id)]))[0].payload["belongs_to_set"]
    ) == {"first", "second"}


async def test_failed_merge_rolls_back_batch_and_staging_can_be_reused(adapter, names):
    name = names[0]
    await adapter.create_collection(name, payload_schema=Point)
    points = [Point(id=UUID(int=1), text="good"), Point(id=UUID(int=2), text="bad")]
    with pytest.raises(Exception, match="dimensions"):
        await adapter.create_data_points(name, points, use_copy=True)
    assert await rows(adapter, name) == []
    await adapter.create_data_points(name, [points[0]], use_copy=True)
    assert len(await rows(adapter, name)) == 1


async def test_concurrent_overlapping_batches_keep_all_tags(adapter, names):
    name = names[0]
    points = [Point(text=str(i)) for i in range(20)]
    await adapter.create_collection(name, payload_schema=Point)
    second = PGVectorAdapter(URL, None, Embeddings())
    try:
        await asyncio.gather(
            *(
                (adapter if i % 2 else second).create_data_points(
                    name,
                    [
                        p.model_copy(update={"belongs_to_set": [str(i)]})
                        for p in (points if i % 2 else list(reversed(points)))
                    ],
                    use_copy=True,
                )
                for i in range(8)
            )
        )
        stored = await adapter.retrieve(name, [str(p.id) for p in points])
        assert len(stored) == len(points)
        assert all(set(p.payload["belongs_to_set"]) == {str(i) for i in range(8)} for p in stored)
    finally:
        await second.close()


async def test_copy_pins_target_schema_and_leaves_public_untouched(adapter, names):
    name = names[0]
    schema = "copy_" + uuid4().hex[:12]
    await adapter.create_data_points(name, [Point(text="public")])
    before = await rows(adapter, name)
    async with adapter.engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    pinned = PGVectorAdapter(URL, None, Embeddings(), schema=schema)
    try:
        point = Point(text="private")
        await pinned.create_data_points(name, [point, point], use_copy=True)
        assert len(await pinned.retrieve(name, [str(point.id)])) == 1
        assert await rows(adapter, name) == before
    finally:
        await pinned.close()
        async with adapter.engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
