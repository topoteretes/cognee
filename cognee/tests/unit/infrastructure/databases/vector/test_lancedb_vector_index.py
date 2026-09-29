"""LanceDB collections get a vector index once they are large enough.

Without an index every search reads every stored vector. The adapter builds a
product-quantized index when a collection reaches LANCEDB_VECTOR_INDEX_MIN_ROWS
and re-ranks indexed searches on the full vectors.
"""

import random
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

COLLECTION = "Indexed_text"
DIMENSIONS = 32


class _Embeddings:
    def get_vector_size(self):
        return DIMENSIONS

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        raise AssertionError("raw vector upserts and vector queries must not embed")


class _Payload(BaseModel):
    text: str


def _adapter(tmp_path, monkeypatch, *, min_rows, subprocess=False, refine_factor=10):
    monkeypatch.setenv("LANCEDB_VECTOR_INDEX_MIN_ROWS", str(min_rows))
    monkeypatch.setenv("LANCEDB_VECTOR_INDEX_REFINE_FACTOR", str(refine_factor))
    if subprocess:
        return LanceDBAdapter.create_subprocess(str(tmp_path / "db"), None, _Embeddings())
    return LanceDBAdapter(str(tmp_path / "db"), None, _Embeddings())


def _rows(count, seed=0):
    """Rows scattered around a few centers, the way embeddings of related texts are."""
    rng = random.Random(seed)
    centers = [[rng.gauss(0, 1) for _ in range(DIMENSIONS)] for _ in range(10)]
    return [
        {
            "id": str(uuid4()),
            "vector": [value + rng.gauss(0, 0.3) for value in rng.choice(centers)],
            "payload": {"text": str(index)},
        }
        for index in range(count)
    ]


async def _write(adapter, rows):
    await adapter.upsert_raw_vectors(COLLECTION, rows, payload_schema=_Payload)


async def _vector_indices(adapter):
    table = await adapter.get_collection(COLLECTION)
    return [index for index in await table.list_indices() if "vector" in index.columns]


async def _found_of_true_nearest(adapter, rows):
    """How many of the 10 true nearest rows a limit-10 search returns, over 20 queries."""
    found = 0
    for row in rows[:20]:
        query = [value + 0.01 for value in row["vector"]]
        exact = await adapter.search(COLLECTION, query_vector=query, limit=None)
        results = await adapter.search(COLLECTION, query_vector=query, limit=10)
        assert str(results[0].id) == row["id"]
        found += len({result.id for result in results} & {result.id for result in exact[:10]})
    return found


@pytest.mark.asyncio
@pytest.mark.parametrize("subprocess", [False, True], ids=["local", "subprocess"])
async def test_large_collection_is_indexed_and_searches_stay_accurate(
    tmp_path, monkeypatch, subprocess
):
    adapter = _adapter(tmp_path, monkeypatch, min_rows=2000, subprocess=subprocess)
    rows = _rows(2000)
    try:
        await _write(adapter, rows)

        indices = await _vector_indices(adapter)
        assert [index.index_type for index in indices] == ["IvfPq"]
        # Measured 91-96% over repeated runs; index training is not seeded.
        assert await _found_of_true_nearest(adapter, rows) >= 0.8 * 200
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_reranking_finds_more_of_the_true_nearest_rows(tmp_path, monkeypatch):
    rows = _rows(2000)
    adapter = _adapter(tmp_path, monkeypatch, min_rows=2000)
    try:
        await _write(adapter, rows)
        reranked = await _found_of_true_nearest(adapter, rows)
    finally:
        await adapter.close()

    adapter = _adapter(tmp_path, monkeypatch, min_rows=2000, refine_factor=1)
    try:
        assert len(await _vector_indices(adapter)) == 1
        assert reranked > await _found_of_true_nearest(adapter, rows)
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_small_collection_is_not_indexed(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch, min_rows=2000)
    try:
        await _write(adapter, _rows(1999))
        assert await _vector_indices(adapter) == []
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_collection_is_indexed_when_it_grows_past_the_threshold(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch, min_rows=600)
    try:
        with patch.object(LanceDBAdapter, "VECTOR_INDEX_CHECK_EVERY_N_WRITES", 2):
            await _write(adapter, _rows(300, seed=1))
            await _write(adapter, _rows(300, seed=2))
            # 600 rows, but this write falls between two checks.
            assert await _vector_indices(adapter) == []

            await _write(adapter, _rows(1, seed=3))
            assert len(await _vector_indices(adapter)) == 1
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_indexing_can_be_disabled(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch, min_rows=0)
    try:
        await _write(adapter, _rows(2000))
        assert await _vector_indices(adapter) == []
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_threshold_below_training_minimum_waits_for_enough_rows(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch, min_rows=10)
    try:
        with patch.object(LanceDBAdapter, "VECTOR_INDEX_CHECK_EVERY_N_WRITES", 1):
            await _write(adapter, _rows(255))
            assert await _vector_indices(adapter) == []

            await _write(adapter, _rows(1, seed=1))
            assert len(await _vector_indices(adapter)) == 1
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_existing_index_is_kept(tmp_path, monkeypatch):
    from lancedb.index import IvfFlat

    adapter = _adapter(tmp_path, monkeypatch, min_rows=0)
    try:
        await _write(adapter, _rows(500))
        table = await adapter.get_collection(COLLECTION)
        await table.create_index("vector", config=IvfFlat(distance_type="cosine", num_partitions=2))
    finally:
        await adapter.close()

    adapter = _adapter(tmp_path, monkeypatch, min_rows=100)
    try:
        await _write(adapter, _rows(1, seed=1))
        assert [index.index_type for index in await _vector_indices(adapter)] == ["IvfFlat"]
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_failed_build_does_not_fail_the_write_and_is_not_repeated(tmp_path, monkeypatch):
    import lancedb

    builds = []

    async def _broken_create_index(self, column, **kwargs):
        builds.append(column)
        raise RuntimeError("out of memory")

    adapter = _adapter(tmp_path, monkeypatch, min_rows=300)
    monkeypatch.setattr(lancedb.AsyncTable, "create_index", _broken_create_index)
    try:
        with patch.object(LanceDBAdapter, "VECTOR_INDEX_CHECK_EVERY_N_WRITES", 1):
            rows = _rows(300)
            await _write(adapter, rows)
            await _write(adapter, _rows(1, seed=1))

        assert builds == ["vector"]
        stored = await adapter.retrieve(COLLECTION, [row["id"] for row in rows])
        assert len(stored) == 300
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_search_without_limit_scores_every_row_of_an_indexed_collection(
    tmp_path, monkeypatch
):
    adapter = _adapter(tmp_path, monkeypatch, min_rows=1000)
    rows = _rows(1000)
    try:
        await _write(adapter, rows)
        assert len(await _vector_indices(adapter)) == 1

        results = await adapter.search(COLLECTION, query_vector=rows[0]["vector"], limit=None)

        assert len(results) == 1000
        assert str(results[0].id) == rows[0]["id"]
        assert results[0].score == pytest.approx(0.0, abs=1e-5)
        scores = [result.score for result in results]
        assert scores == sorted(scores)
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_rows_written_after_the_build_are_found(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch, min_rows=500)
    try:
        await _write(adapter, _rows(500))
        assert len(await _vector_indices(adapter)) == 1

        late = _rows(1, seed=7)
        await _write(adapter, late)

        results = await adapter.search(COLLECTION, query_vector=late[0]["vector"], limit=3)
        assert str(results[0].id) == late[0]["id"]
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_worker_builds_only_known_index_configs():
    from cognee_db_workers.harness import HandleRegistry, Request
    from cognee_db_workers.lancedb_protocol import OP_TABLE_CREATE_INDEX
    from cognee_db_workers.lancedb_worker import DISPATCH

    class _Table:
        async def create_index(self, column, *, config):
            raise AssertionError("must not be reached")

    registry = HandleRegistry()
    handle_id = registry.register(_Table())

    with pytest.raises(ValueError, match="Unsupported index config"):
        await DISPATCH[OP_TABLE_CREATE_INDEX](
            registry,
            Request(op=OP_TABLE_CREATE_INDEX, handle_id=handle_id, args=("vector", "FTS", {})),
        )
