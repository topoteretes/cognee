from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.infrastructure.engine import DataPoint

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


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_lancedb_upsert_raw_vectors_without_schema_stores_index_rows(tmp_path):
    """No payload schema: the row is an ``IndexSchema`` row, as ``index_data_points`` writes
    it, so a row copied from a ``<Type>_<field>`` collection keeps that shape."""
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import IndexSchema

    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    point_id, document_id = uuid4(), uuid4()

    await adapter.upsert_raw_vectors(
        "DatasetRoute_text",
        [
            {
                "id": point_id,
                "vector": [1.0, 0.0, 0.0],
                "payload": {
                    "id": str(point_id),
                    "text": "copied",
                    "document_id": str(document_id),
                    "belongs_to_set": ["proj"],
                },
            }
        ],
    )

    [row] = await adapter.retrieve("DatasetRoute_text", [str(point_id)])
    assert row.payload["text"] == "copied"
    assert row.payload["document_id"] == str(document_id)
    assert row.payload["belongs_to_set"] == ["proj"]
    # The same columns as an ingested index row, in the same order.
    expected = adapter.get_data_point_schema(IndexSchema).model_fields
    assert list(row.payload) == list(expected)


class _OldRoutingRow(DataPoint):
    """The routing row shape before it became an index row (cognee 1.6.6)."""

    text: str
    dataset_id: str
    is_root: bool = False
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_lancedb_upsert_raw_vectors_migrates_a_table_of_an_older_shape(tmp_path):
    """A table created with an older payload shape is rebuilt to the index row shape on
    the first write, keeping its rows, the way ``create_data_points`` does."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    collection = "DatasetRoute_text"
    old_id, new_id = uuid4(), uuid4()

    await adapter.upsert_raw_vectors(
        collection,
        [
            {
                "id": old_id,
                "vector": [1.0, 0.0, 0.0],
                "payload": {
                    "id": str(old_id),
                    "text": "old root",
                    "dataset_id": "d",
                    "is_root": True,
                },
            }
        ],
        payload_schema=_OldRoutingRow,
    )

    await adapter.upsert_raw_vectors(
        collection,
        [
            {
                "id": new_id,
                "vector": [0.0, 1.0, 0.0],
                "payload": {"id": str(new_id), "text": "bucket", "document_id": "doc"},
            }
        ],
    )

    rows = {
        str(row.id): row.payload
        for row in await adapter.retrieve(collection, [str(old_id), str(new_id)])
    }
    assert rows[str(old_id)]["text"] == "old root"
    assert rows[str(new_id)]["document_id"] == "doc"
    assert "is_root" not in rows[str(old_id)]
    assert set(rows[str(old_id)]) == set(rows[str(new_id)])


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_concurrent_first_writes_after_an_upgrade_keep_every_row(tmp_path):
    """The routing table is one table for every dataset, so two datasets' first
    writes after an upgrade both find the older shape and both migrate. The
    second migration must not rebuild the table from a copy taken before the
    first one wrote its row: the copy and the replacement run under the write
    lock, and a migration that gets the lock after another one rebuilt the
    table writes its row into that table instead."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    collection = "DatasetRoute_text"
    old_id, first_id, second_id = uuid4(), uuid4(), uuid4()

    await adapter.upsert_raw_vectors(
        collection,
        [
            {
                "id": old_id,
                "vector": [1.0, 0.0, 0.0],
                "payload": {"id": str(old_id), "text": "old root", "dataset_id": "d"},
            }
        ],
        payload_schema=_OldRoutingRow,
    )

    def new_row(row_id, text):
        return {
            "id": row_id,
            "vector": [0.0, 1.0, 0.0],
            "payload": {"id": str(row_id), "text": text, "document_id": "doc"},
        }

    await asyncio.gather(
        adapter.upsert_raw_vectors(collection, [new_row(first_id, "first dataset")]),
        adapter.upsert_raw_vectors(collection, [new_row(second_id, "second dataset")]),
    )

    rows = {
        str(row.id): row.payload["text"]
        for row in await adapter.retrieve(collection, [str(old_id), str(first_id), str(second_id)])
    }
    assert rows == {
        str(old_id): "old root",
        str(first_id): "first dataset",
        str(second_id): "second dataset",
    }
