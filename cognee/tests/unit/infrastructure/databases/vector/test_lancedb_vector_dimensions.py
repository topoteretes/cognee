from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.exceptions import CogneeApiError
from cognee.infrastructure.databases.vector.exceptions import VectorDimensionMismatchError
from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
    IndexSchema,
    LanceDBAdapter,
)


class _EmbeddingEngine:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.embed_calls = 0

    def get_vector_size(self):
        return self.dimension

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        self.embed_calls += 1
        return [[0.1] * self.dimension for _ in texts]


def _point(text: str) -> IndexSchema:
    return IndexSchema(id=str(uuid4()), text=text)


@pytest.mark.asyncio
async def test_dimension_mismatch_raises_actionable_cognee_error_before_embedding(tmp_path):
    original_engine = _EmbeddingEngine(3)
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=original_engine,
    )
    collection_name = "Test_text"
    await adapter.create_collection(collection_name, IndexSchema)
    await adapter.create_data_points(collection_name, [_point("existing")])

    replacement_engine = _EmbeddingEngine(4)
    adapter.embedding_engine = replacement_engine

    with pytest.raises(VectorDimensionMismatchError) as raised:
        await adapter.create_data_points(collection_name, [_point("incoming")])

    assert isinstance(raised.value, CogneeApiError)
    assert raised.value.status_code == 422
    assert replacement_engine.embed_calls == 0
    assert collection_name in raised.value.message
    assert "3-dimensional" in raised.value.message
    assert "4-dimensional" in raised.value.message
    assert "Re-create the collection or migrate its vectors" in raised.value.message
    assert await (await adapter.get_collection(collection_name)).count_rows() == 1


@pytest.mark.asyncio
async def test_matching_dimension_preserves_existing_upsert_behavior(tmp_path):
    engine = _EmbeddingEngine(3)
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=engine,
    )
    collection_name = "Test_text"

    await adapter.create_collection(collection_name, IndexSchema)
    await adapter.create_data_points(collection_name, [_point("first")])
    await adapter.create_data_points(collection_name, [_point("second")])

    assert engine.embed_calls == 2
    assert await (await adapter.get_collection(collection_name)).count_rows() == 2


@pytest.mark.asyncio
async def test_subprocess_dimension_check_reads_schema_without_materializing_rows(
    tmp_path, monkeypatch
):
    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import (
        RemoteLanceDBTable,
    )

    original_engine = _EmbeddingEngine(3)
    adapter = LanceDBAdapter.create_subprocess(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=original_engine,
    )
    collection_name = "Test_text"

    try:
        await adapter.create_collection(collection_name, IndexSchema)
        await adapter.create_data_points(collection_name, [_point("existing")])

        async def fail_if_materialized(_self):
            raise AssertionError("dimension validation must not materialize table rows")

        monkeypatch.setattr(RemoteLanceDBTable, "to_arrow", fail_if_materialized)

        replacement_engine = _EmbeddingEngine(4)
        adapter.embedding_engine = replacement_engine

        with pytest.raises(VectorDimensionMismatchError):
            await adapter.create_data_points(collection_name, [_point("incoming")])

        assert replacement_engine.embed_calls == 0
    finally:
        await adapter.close()
