"""The first ``create_data_points`` on a collection creates it without deadlocking.

``create_collection`` takes the adapter's write lock itself; a caller that also
held it around the call (as ``create_data_points`` once did) waited forever on
the non-reentrant lock. Every pipeline write goes through a prior explicit
``create_collection``, so the hang only showed up for a direct first write, such
as the dataset routing mirror.
"""

from __future__ import annotations

import asyncio

import pytest

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
        return [[1.0, 0.0, 0.0] for _ in texts]


class _Row(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_first_write_to_a_missing_collection_creates_it_and_returns(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    rows = [_Row(text="alpha"), _Row(text="beta")]

    await asyncio.wait_for(adapter.create_data_points("FirstWrite_text", rows), timeout=30)

    stored = await adapter.retrieve("FirstWrite_text", [str(row.id) for row in rows])
    assert {str(row.id) for row in stored} == {str(row.id) for row in rows}
