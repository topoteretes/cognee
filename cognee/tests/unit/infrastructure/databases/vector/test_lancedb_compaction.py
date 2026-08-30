from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel

try:
    from cognee.infrastructure.databases.vector.config import get_vectordb_config
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _Payload(BaseModel):
    slot: int
    label: str


def _fragment_count(db_path: str, collection_name: str) -> int:
    """Count on-disk data fragment files for a table.

    Mirrors the reproduction command from
    https://github.com/topoteretes/cognee/issues/4684:
    `find ... -path "*.lance/data/*" -type f | wc -l`
    -- counting files directly rather than depending on a specific
    lancedb version's stats API.
    """
    data_dir = Path(db_path) / f"{collection_name}.lance" / "data"
    if not data_dir.exists():
        return 0
    return len(list(data_dir.iterdir()))


async def _write_n_points(adapter: LanceDBAdapter, collection: str, n: int) -> None:
    for i in range(n):
        await adapter.upsert_raw_vectors(
            collection,
            [
                {
                    "id": uuid4(),
                    "vector": [0.1, 0.2, 0.3],
                    "payload": {"slot": i, "label": f"row-{i}"},
                }
            ],
            payload_schema=_Payload,
        )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_lancedb_compacts_after_configured_write_interval(tmp_path, monkeypatch):
    """Fragments must be folded back down once the configured write count is hit."""
    monkeypatch.setenv("VECTOR_DB_COMPACTION_WRITE_INTERVAL", "3")
    get_vectordb_config.cache_clear()
    try:
        db_path = str(tmp_path / "db")
        adapter = LanceDBAdapter(
            url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine()
        )
        collection = "CompactionTarget_label"

        await _write_n_points(adapter, collection, 3)

        # Without compaction this would be 3 fragments (one per upsert, per
        # the issue's own measurement). optimize() should have folded them
        # into (at most) one on the 3rd write.
        assert _fragment_count(db_path, collection) <= 1
    finally:
        get_vectordb_config.cache_clear()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_lancedb_compaction_disabled_when_interval_is_zero(tmp_path, monkeypatch):
    """Interval 0 must restore today's uncompacted behaviour exactly."""
    monkeypatch.setenv("VECTOR_DB_COMPACTION_WRITE_INTERVAL", "0")
    get_vectordb_config.cache_clear()
    try:
        db_path = str(tmp_path / "db")
        adapter = LanceDBAdapter(
            url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine()
        )
        collection = "NoCompactionTarget_label"

        await _write_n_points(adapter, collection, 6)

        # One fragment per upsert, never compacted -- today's behaviour,
        # preserved for anyone who opts out.
        assert _fragment_count(db_path, collection) == 6
    finally:
        get_vectordb_config.cache_clear()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_lancedb_compaction_does_not_lose_rows(tmp_path, monkeypatch):
    """Compaction must never change row counts or retrievable data (core safety property)."""
    monkeypatch.setenv("VECTOR_DB_COMPACTION_WRITE_INTERVAL", "2")
    get_vectordb_config.cache_clear()
    try:
        db_path = str(tmp_path / "db")
        adapter = LanceDBAdapter(
            url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine()
        )
        collection = "RowSafetyTarget_label"
        ids = [uuid4() for _ in range(5)]

        for i, point_id in enumerate(ids):
            await adapter.upsert_raw_vectors(
                collection,
                [
                    {
                        "id": point_id,
                        "vector": [0.1, 0.2, 0.3],
                        "payload": {"slot": i, "label": f"row-{i}"},
                    }
                ],
                payload_schema=_Payload,
            )

        table = await adapter.get_collection(collection)
        assert await table.count_rows() == 5

        retrieved = await adapter.retrieve(collection, [str(pid) for pid in ids])
        assert len(retrieved) == 5
    finally:
        get_vectordb_config.cache_clear()