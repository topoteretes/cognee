"""``delete_collection`` drops the table, so a new vector width can follow.

A LanceDB table's vector column is fixed to the width it was created with.
``prune_system`` uses ``delete_collection`` on the dataset routing collection
for this reason: emptied rather than dropped, the table would reject every
write after an ``EMBEDDING_MODEL`` change.
"""

from __future__ import annotations

import pytest

from cognee.infrastructure.engine import DataPoint

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    def __init__(self, width):
        self.width = width

    def get_vector_size(self):
        return self.width

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[1.0] + [0.0] * (self.width - 1) for _ in texts]


class _Row(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_delete_collection_drops_the_table_and_a_wider_model_can_recreate_it(tmp_path):
    url = str(tmp_path / "db")
    narrow = LanceDBAdapter(url=url, api_key=None, embedding_engine=_FakeEmbeddingEngine(3))
    await narrow.create_data_points("Routes_text", [_Row(text="alpha")])
    assert await narrow.get_stored_vector_size() == 3

    await narrow.delete_collection("Routes_text")

    assert await narrow.has_collection("Routes_text") is False
    await narrow.delete_collection("Routes_text")  # a missing table is not an error

    wide = LanceDBAdapter(url=url, api_key=None, embedding_engine=_FakeEmbeddingEngine(4))
    await wide.create_data_points("Routes_text", [_Row(text="beta")])
    assert await wide.get_stored_vector_size() == 4
