"""Backend capability tests for graph-native vector cleanup on the default
vector backend (LanceDB) — COG-5522 Part 1.

Graph-native delete computes vector ids from graph snapshots and hands them to
``delete_data_points``. Those ids may be UUIDs or graph-computed deterministic
strings, and the delete must be idempotent (missing collection / missing id /
empty input are all no-ops). ``remove_belongs_to_set_tags`` branch coverage
lives in ``test_belongs_to_set_merge.py``; this module pins the delete path.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        IndexSchema,
        LanceDBAdapter,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed"),
]


class _FakeEmbeddingEngine:
    """Deterministic stub embedding engine; avoids external API calls."""

    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _new_adapter(tmp_path):
    return LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )


async def _seed(adapter, collection, ids):
    await adapter.create_collection(collection, IndexSchema)
    await adapter.create_data_points(
        collection,
        [IndexSchema(id=str(i), text=f"text-{i}") for i in ids],
    )


async def test_delete_data_points_deletes_uuid_and_string_ids(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        collection = "Entity_name"
        uuid_id = str(uuid4())
        string_id = "germany_knows_alice"  # graph-computed deterministic id form
        keep_id = str(uuid4())
        await _seed(adapter, collection, [uuid_id, string_id, keep_id])

        await adapter.delete_data_points(collection, [uuid_id, string_id])

        remaining = await adapter.retrieve(collection, [uuid_id, string_id, keep_id])
        assert {str(r.id) for r in remaining} == {keep_id}
    finally:
        await adapter.close()


async def test_delete_data_points_is_idempotent(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        collection = "Entity_name"
        point_id = str(uuid4())
        await _seed(adapter, collection, [point_id])

        # Missing id on an existing collection: no error, no-op.
        await adapter.delete_data_points(collection, [str(uuid4())])
        assert len(await adapter.retrieve(collection, [point_id])) == 1

        # Delete, then re-delete the same id: idempotent.
        await adapter.delete_data_points(collection, [point_id])
        await adapter.delete_data_points(collection, [point_id])
        assert await adapter.retrieve(collection, [point_id]) == []
    finally:
        await adapter.close()


async def test_delete_data_points_missing_collection_is_noop(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        # No collection created at all — must not raise.
        await adapter.delete_data_points("NeverCreated_name", [str(uuid4())])
    finally:
        await adapter.close()


async def test_delete_data_points_empty_ids_is_noop(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        collection = "Entity_name"
        point_id = str(uuid4())
        await _seed(adapter, collection, [point_id])

        await adapter.delete_data_points(collection, [])

        assert len(await adapter.retrieve(collection, [point_id])) == 1
    finally:
        await adapter.close()


async def test_delete_data_points_escapes_single_quotes(tmp_path):
    adapter = _new_adapter(tmp_path)
    try:
        collection = "Entity_name"
        quoted_id = "o'brien"  # would break an unescaped id = '...' predicate
        keep_id = str(uuid4())
        await _seed(adapter, collection, [quoted_id, keep_id])

        await adapter.delete_data_points(collection, [quoted_id])

        remaining = await adapter.retrieve(collection, [quoted_id, keep_id])
        assert {str(r.id) for r in remaining} == {keep_id}
    finally:
        await adapter.close()
