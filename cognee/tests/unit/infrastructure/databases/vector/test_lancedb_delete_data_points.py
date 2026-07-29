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
        raise AssertionError("delete tests must not call embed_text")


class _Payload(BaseModel):
    label: str


def _make_adapter(tmp_path) -> LanceDBAdapter:
    return LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )


async def _seed(adapter: LanceDBAdapter, collection: str, ids: list) -> None:
    await adapter.upsert_raw_vectors(
        collection,
        [
            {
                "id": point_id,
                "vector": [1.0, 0.0, 0.0],
                "payload": {"label": f"row_{index}"},
            }
            for index, point_id in enumerate(ids)
        ],
        payload_schema=_Payload,
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_delete_data_points_removes_only_matching_rows(tmp_path):
    adapter = _make_adapter(tmp_path)
    collection = "DeleteTarget_vector"
    ids = [uuid4() for _ in range(6)]
    await _seed(adapter, collection, ids)

    doomed, survivors = ids[:4], ids[4:]
    # Non-existent ids (and a quote-bearing id exercising predicate escaping)
    # must no-op rather than raise or over-delete.
    await adapter.delete_data_points(collection, [*doomed, uuid4(), "o'brien"])

    remaining = await adapter.retrieve(collection, [str(point_id) for point_id in ids])
    remaining_ids = {str(row.id) for row in remaining}
    assert remaining_ids == {str(point_id) for point_id in survivors}


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_delete_data_points_batches_predicates(tmp_path, monkeypatch):
    """Regression: a large id list must not issue one delete commit per id.

    Every ``collection.delete`` is a LanceDB commit appending a table version;
    per-id commits made a 13k-id wipe degrade into a multi-hour stall during
    ``forget(everything=True)``.
    """
    adapter = _make_adapter(tmp_path)

    predicates: list[str] = []

    class _SpyCollection:
        async def delete(self, predicate: str) -> None:
            predicates.append(predicate)

    async def _has_collection(_name):
        return True

    async def _get_collection(_name):
        return _SpyCollection()

    monkeypatch.setattr(adapter, "has_collection", _has_collection)
    monkeypatch.setattr(adapter, "get_collection", _get_collection)

    batch_size = LanceDBAdapter.DELETE_PREDICATE_BATCH_SIZE
    ids = [uuid4() for _ in range(2 * batch_size + 500)]
    await adapter.delete_data_points("AnyCollection", ids)

    assert len(predicates) == 3
    assert all(predicate.startswith("id IN (") for predicate in predicates)
    # Every id appears exactly once across the batched predicates.
    joined = "".join(predicates)
    assert all(f"'{point_id}'" in joined for point_id in map(str, ids))
