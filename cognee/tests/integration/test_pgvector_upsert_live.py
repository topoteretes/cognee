"""pgvector ``ON CONFLICT`` upserts against a live Postgres: skip, re-embed, merge.

The conflict update of ``create_data_points`` runs only when the payload differs from the
stored one apart from ``created_at``/``updated_at``, and takes the new vector only when
the row's text changed. These tests run that SQL: an identical re-upsert leaves the row
version untouched (same ``ctid`` and ``xmin``), a changed text replaces the vector, and a
new ``belongs_to_set`` tag is still merged in.

Rows are written as cognee writes them, through the adapter's ``index_data_points``. No LLM
or embedding calls: the embedding engine is a stub whose vectors differ on every call, so a
kept vector is told apart from a replaced one. Skipped unless ``TEST_PGVECTOR_URL`` is set,
e.g.:

    TEST_PGVECTOR_URL="postgresql+asyncpg://cognee:cognee@127.0.0.1:5432/cognee_db"
"""

from __future__ import annotations

import asyncio
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
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
    )
except ModuleNotFoundError:  # postgres extra not installed
    pytestmark.append(pytest.mark.skip(reason="pgvector extra not installed"))

DIMENSIONS = 8


class _Embeddings:
    """Embeddings that differ on every call, even for the same text, so a test can tell
    which write's vector a row holds. ``embedded[text]`` lists them in call order."""

    def __init__(self):
        self.embedded: dict[str, list[list[float]]] = {}

    def get_vector_size(self) -> int:
        return DIMENSIONS

    def get_batch_size(self) -> int:
        return 100

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(value) for value in texts]

    def _embed(self, value: str) -> list[float]:
        calls = self.embedded.setdefault(value, [])
        rng = random.Random(f"{value}:{len(calls)}")
        calls.append([rng.uniform(-1, 1) for _ in range(DIMENSIONS)])
        return calls[-1]


class _Point(DataPoint):
    text: str
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
    name = f"IntegUpsertPoint_{uuid4().hex[:10]}_text"
    yield name
    await adapter.delete_collection(name)


async def _write(adapter: PGVectorAdapter, collection: str, points: list[_Point]) -> None:
    """Write as cognee does: ``IndexSchema`` rows through the adapter's ``index_data_points``."""
    index_name, field_name = collection.rsplit("_", 1)
    await adapter.index_data_points(index_name, field_name, points)


async def _row_version(adapter: PGVectorAdapter, collection: str, point_id: UUID) -> tuple:
    """The row's physical location and inserting transaction: both change on any rewrite."""
    async with adapter.get_async_session() as session:
        result = await session.execute(
            text(f'SELECT ctid::text, xmin::text FROM "{collection}" WHERE id = :id'),
            {"id": point_id},
        )
        return tuple(result.one())


async def _stored(adapter: PGVectorAdapter, collection: str, point_id: UUID) -> dict:
    (row,) = await adapter.retrieve(collection, [str(point_id)], include_vector=True)
    return row.payload


def _assert_vector(stored: list[float], expected: list[float]) -> None:
    assert stored == pytest.approx(expected, abs=1e-6)


async def test_identical_reupsert_writes_no_new_row_version(adapter, collection, embeddings):
    """Only the timestamps differ, so the conflict update is skipped."""
    point = _Point(text="same", belongs_to_set=["A"])
    await _write(adapter, collection, [point])
    before = await _row_version(adapter, collection, point.id)
    stamped = (await _stored(adapter, collection, point.id))["updated_at"]

    await asyncio.sleep(0.01)  # IndexSchema stamps milliseconds: the re-upsert's differ
    await _write(adapter, collection, [point])

    assert await _row_version(adapter, collection, point.id) == before
    stored = await _stored(adapter, collection, point.id)
    assert stored["updated_at"] == stamped
    _assert_vector(stored["vector"], embeddings.embedded["same"][0])


async def test_changed_text_replaces_vector(adapter, collection, embeddings):
    """A row keeping its id across a text change (a re-extracted summary) is re-embedded."""
    point_id = uuid4()
    await _write(adapter, collection, [_Point(id=point_id, text="old")])
    _assert_vector(
        (await _stored(adapter, collection, point_id))["vector"], embeddings.embedded["old"][0]
    )

    await _write(adapter, collection, [_Point(id=point_id, text="new")])

    stored = await _stored(adapter, collection, point_id)
    assert stored["text"] == "new"
    _assert_vector(stored["vector"], embeddings.embedded["new"][-1])


async def test_new_tag_is_merged_and_keeps_vector(adapter, collection, embeddings):
    """A metadata-only change updates the payload, unioning the tags, and keeps the vector
    of the first write although the second write embedded the text anew."""
    point_id = uuid4()
    await _write(adapter, collection, [_Point(id=point_id, text="shared", belongs_to_set=["A"])])

    await _write(adapter, collection, [_Point(id=point_id, text="shared", belongs_to_set=["B"])])

    stored = await _stored(adapter, collection, point_id)
    assert sorted(stored["belongs_to_set"]) == ["A", "B"]
    assert len(embeddings.embedded["shared"]) == 2
    _assert_vector(stored["vector"], embeddings.embedded["shared"][0])
