"""``existing_collections``: the interface default and the LanceDB override.

Both must answer the same question as ``has_collection``, name by name, in the
caller's order. The default asks ``has_collection`` itself; LanceDB answers from
one table listing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
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

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _Point(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
async def test_default_keeps_the_names_has_collection_confirms():
    present = {"Entity_name", "EdgeType_relationship_name"}
    adapter = AsyncMock()
    adapter.has_collection = AsyncMock(side_effect=lambda name: name in present)

    names = ["Missing_text", "EdgeType_relationship_name", "Entity_name", "Other_text"]
    result = await VectorDBInterface.existing_collections(adapter, names)

    assert result == ["EdgeType_relationship_name", "Entity_name"]
    assert sorted(c.args[0] for c in adapter.has_collection.await_args_list) == sorted(names)


@pytest.mark.asyncio
async def test_default_with_no_names_asks_nothing():
    adapter = AsyncMock()
    assert await VectorDBInterface.existing_collections(adapter, []) == []
    adapter.has_collection.assert_not_awaited()


@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
@pytest.mark.asyncio
async def test_lancedb_answers_from_one_listing(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    await adapter.create_collection("Alpha_text", _Point)
    await adapter.create_collection("Beta_text", _Point)

    names = ["Beta_text", "Missing_text", "Alpha_text"]
    assert await adapter.existing_collections(names) == ["Beta_text", "Alpha_text"]
    assert await adapter.existing_collections(["Missing_text"]) == []
    probed = [name for name in names if await adapter.has_collection(name)]
    assert await adapter.existing_collections(names) == probed
