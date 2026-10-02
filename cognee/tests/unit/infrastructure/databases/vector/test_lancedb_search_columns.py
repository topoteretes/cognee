"""LanceDB searches read only the columns a ScoredResult carries.

The stored vector is not part of a search result. Selecting it made every
search decode one full embedding per returned row, which for ``limit=None``
is the whole collection.
"""

from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

COLLECTION = "Searched_text"


class _Embeddings:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        raise AssertionError("raw vector upserts and vector queries must not embed")


class _Payload(BaseModel):
    text: str
    belongs_to_set: list[str] = []


@pytest.fixture
def selected_columns(monkeypatch):
    """Record the columns every vector query of the test asks lancedb for."""
    from lancedb.query import AsyncVectorQuery

    selected = []
    original = AsyncVectorQuery.select

    def _select(self, columns):
        selected.append(list(columns))
        return original(self, columns)

    monkeypatch.setattr(AsyncVectorQuery, "select", _select)
    return selected


async def _adapter_with_rows(tmp_path, subprocess=False):
    create = LanceDBAdapter.create_subprocess if subprocess else LanceDBAdapter
    adapter = create(str(tmp_path / "db"), None, _Embeddings())
    rows = [
        {
            "id": str(uuid4()),
            "vector": [1.0, float(index), 0.0],
            "payload": {
                "text": str(index),
                "belongs_to_set": ["even" if index % 2 == 0 else "odd"],
            },
        }
        for index in range(6)
    ]
    await adapter.upsert_raw_vectors(COLLECTION, rows, payload_schema=_Payload)
    return adapter, rows


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [3, None])
async def test_search_does_not_read_stored_vectors(tmp_path, selected_columns, limit):
    adapter, rows = await _adapter_with_rows(tmp_path)
    try:
        results = await adapter.search(COLLECTION, query_vector=[1.0, 0.0, 0.0], limit=limit)

        assert selected_columns == [["id", "_distance"]]
        assert [str(result.id) for result in results] == [
            row["id"] for row in rows[: limit or len(rows)]
        ]
        assert results[0].score == pytest.approx(0.0, abs=1e-6)
        assert all(result.payload is None for result in results)
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_search_with_payload_reads_the_payload_but_not_the_vector(tmp_path, selected_columns):
    adapter, rows = await _adapter_with_rows(tmp_path)
    try:
        results = await adapter.search(
            COLLECTION,
            query_vector=[1.0, 0.0, 0.0],
            limit=2,
            include_payload=True,
            node_name=["odd"],
        )

        assert selected_columns == [["id", "payload", "_distance"]]
        assert [str(result.id) for result in results] == [rows[1]["id"], rows[3]["id"]]
        assert [result.payload["text"] for result in results] == ["1", "3"]
        assert all("vector" not in result.payload for result in results)
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_search_in_subprocess_mode_returns_the_same_results(tmp_path):
    adapter, rows = await _adapter_with_rows(tmp_path, subprocess=True)
    try:
        results = await adapter.search(
            COLLECTION, query_vector=[1.0, 0.0, 0.0], limit=None, include_payload=True
        )

        assert [str(result.id) for result in results] == [row["id"] for row in rows]
        assert [result.payload["text"] for result in results] == [str(i) for i in range(6)]
    finally:
        await adapter.close()
