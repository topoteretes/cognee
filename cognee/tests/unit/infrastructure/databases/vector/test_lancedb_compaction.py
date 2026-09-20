from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import AsyncMock, Mock
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
        adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
        collection = "CompactionTarget_label"

        await _write_n_points(adapter, collection, 2)
        table = await adapter.get_collection(collection)
        assert (await table.stats())["fragment_stats"]["num_fragments"] == 2
        await _write_n_points(adapter, collection, 1)

        # Count the current manifest's fragments: older files must remain
        # within LanceDB's retention window for readers on recent snapshots.
        table = await adapter.get_collection(collection)
        assert (await table.stats())["fragment_stats"]["num_fragments"] == 1
        assert await table.count_rows() == 3
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
        adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
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
        adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
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


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_compaction_preserves_a_recent_reader_snapshot(tmp_path, monkeypatch):
    import lancedb

    monkeypatch.setenv("VECTOR_DB_COMPACTION_WRITE_INTERVAL", "3")
    get_vectordb_config.cache_clear()
    try:
        db_path = str(tmp_path / "db")
        adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
        collection = "SnapshotTarget_label"
        await _write_n_points(adapter, collection, 1)
        connection = await lancedb.connect_async(db_path)
        reader = await connection.open_table(collection)
        await reader.checkout(await reader.version())
        expected = (await reader.to_arrow()).to_pylist()

        # Two more writes trigger maintenance while another reader still owns
        # the preceding version. Compaction must not delete that reader's files.
        await _write_n_points(adapter, collection, 2)

        assert (await reader.to_arrow()).to_pylist() == expected
        current = await adapter.get_collection(collection)
        assert await current.count_rows() == 3
    finally:
        get_vectordb_config.cache_clear()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_compaction_reaches_the_subprocess_worker(tmp_path, monkeypatch):
    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import RemoteLanceDBTable
    from cognee_db_workers.lancedb_protocol import OP_TABLE_OPTIMIZE

    monkeypatch.setenv("VECTOR_DB_COMPACTION_WRITE_INTERVAL", "1")
    get_vectordb_config.cache_clear()
    try:
        adapter = LanceDBAdapter(
            url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
        )
        session = Mock()
        session.call_async = AsyncMock()
        table = RemoteLanceDBTable(session, 17, "WorkerTarget_label")

        await adapter._maybe_compact(table.name, table)

        session.call_async.assert_awaited_once()
        request = session.call_async.await_args.args[0]
        assert request.op == OP_TABLE_OPTIMIZE
        assert request.handle_id == 17
        assert request.args == ()
    finally:
        get_vectordb_config.cache_clear()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_compaction_setting_keeps_vector_factory_usable(tmp_path, monkeypatch):
    from cognee.infrastructure.databases.vector.config import VectorConfig

    factory = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")
    monkeypatch.setattr(factory, "get_embedding_engine", _FakeEmbeddingEngine)
    config = VectorConfig(
        vector_db_provider="lancedb",
        vector_db_url=str(tmp_path / "db"),
        vector_db_name="factory_compaction",
        vector_db_subprocess_enabled=False,
        vector_db_compaction_write_interval=2,
    )

    adapter = factory.create_vector_engine(**config.to_dict())
    try:
        await _write_n_points(adapter, "FactoryTarget_label", 1)
        table = await adapter.get_collection("FactoryTarget_label")
        assert await table.count_rows() == 1
    finally:
        await adapter.close()
