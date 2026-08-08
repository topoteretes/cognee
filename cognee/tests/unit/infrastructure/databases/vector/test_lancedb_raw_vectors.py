from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, _texts):
        raise AssertionError("raw vector upsert must not call embed_text")


class _RawPayload(BaseModel):
    slot: int
    label: str
    centroid: list[float]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_lancedb_upsert_raw_vectors_writes_and_updates_payload(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    collection = "RawCentroid_vector"
    point_id = uuid4()

    await adapter.upsert_raw_vectors(
        collection,
        [
            {
                "id": point_id,
                "vector": [1.0, 0.0, 0.0],
                "payload": {"slot": 0, "label": "first", "centroid": [1.0, 0.0, 0.0]},
            }
        ],
        payload_schema=_RawPayload,
    )
    await adapter.upsert_raw_vectors(
        collection,
        [
            {
                "id": point_id,
                "vector": [0.0, 1.0, 0.0],
                "payload": {"slot": 0, "label": "updated", "centroid": [0.0, 1.0, 0.0]},
            }
        ],
        payload_schema=_RawPayload,
    )

    retrieved = await adapter.retrieve(collection, [str(point_id)])
    assert len(retrieved) == 1
    assert retrieved[0].payload["label"] == "updated"
    assert retrieved[0].payload["centroid"] == [0.0, 1.0, 0.0]

    results = await adapter.search(
        collection,
        query_vector=[0.0, 1.0, 0.0],
        limit=1,
        include_payload=True,
    )
    assert str(results[0].id) == str(point_id)
    assert results[0].payload["label"] == "updated"
