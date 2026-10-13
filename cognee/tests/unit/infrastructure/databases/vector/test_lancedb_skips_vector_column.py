"""LanceDB search() never selects the vector column.

Results carry only id, payload and distance. Selecting the vector column made a
limit=None search (node-set filtered retrieval reads whole collections) load
every stored embedding.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint

try:
    import lancedb.query

    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

pytestmark = pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb is not installed")


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3 + i / 10] for i, _ in enumerate(texts)]


class _TaggedPoint(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


async def _seeded_adapter(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    points = [
        _TaggedPoint(id=str(uuid4()), text=f"point {i}", belongs_to_set=["A" if i % 2 else "B"])
        for i in range(4)
    ]
    await adapter.create_collection("Tagged_text", _TaggedPoint)
    await adapter.create_data_points("Tagged_text", points)
    return adapter, points


@pytest.mark.asyncio
@pytest.mark.parametrize("include_payload", [False, True])
@pytest.mark.parametrize("limit", [None, 2])
@pytest.mark.parametrize("node_name", [None, ["A"]])
async def test_search_never_selects_vector_column(
    tmp_path, monkeypatch, include_payload, limit, node_name
):
    adapter, points = await _seeded_adapter(tmp_path)

    selected = []
    original_select = lancedb.query.AsyncQueryBase.select

    def _recording_select(self, columns):
        selected.append(list(columns))
        return original_select(self, columns)

    monkeypatch.setattr(lancedb.query.AsyncQueryBase, "select", _recording_select)

    results = await adapter.search(
        "Tagged_text",
        query_vector=[0.1, 0.2, 0.3],
        limit=limit,
        include_payload=include_payload,
        node_name=node_name,
    )

    assert selected, "search() should project columns explicitly"
    assert all("vector" not in columns for columns in selected)
    expected_count = 2 if node_name else (limit or len(points))
    assert len(results) == expected_count
    if include_payload:
        assert all(result.payload["text"].startswith("point") for result in results)
    else:
        assert all(result.payload is None for result in results)
