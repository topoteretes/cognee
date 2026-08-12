"""Collection replacement contracts shared by vector adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint

lancedb = pytest.importorskip("lancedb")


class _FakeEmbeddingEngine:
    """Local deterministic embeddings keep this contract test offline."""

    def get_vector_size(self) -> int:
        return 3

    def get_batch_size(self) -> int:
        return 100

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 1.0] for text in texts]


class _IndexedPoint(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


def _adapter(path):
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    return LanceDBAdapter(str(path), None, _FakeEmbeddingEngine())


async def _batches(*batches: list[DataPoint]) -> AsyncIterator[list[DataPoint]]:
    for batch in batches:
        yield batch


async def _interrupted_batches(batch: list[DataPoint]) -> AsyncIterator[list[DataPoint]]:
    yield batch
    raise RuntimeError("embedding stream interrupted")


@pytest.mark.asyncio
async def test_replace_index_data_points_replaces_only_named_collection(tmp_path):
    """Replacement must clear stale rows only from its requested collection."""
    adapter = _adapter(tmp_path / "vectors")
    old_id, replacement_id, triplet_id = (str(uuid4()) for _ in range(3))

    await adapter.index_data_points("EdgeInstance", "text", [_IndexedPoint(id=old_id, text="old")])
    await adapter.index_data_points("Triplet", "text", [_IndexedPoint(id=triplet_id, text="keep")])

    await adapter.replace_index_data_points(
        "EdgeInstance",
        "text",
        _batches([_IndexedPoint(id=replacement_id, text="replacement")]),
    )

    assert await adapter.retrieve("EdgeInstance_text", [old_id]) == []
    replacement = await adapter.retrieve("EdgeInstance_text", [replacement_id])
    assert replacement[0].payload["text"] == "replacement"
    triplet = await adapter.retrieve("Triplet_text", [triplet_id])
    assert triplet[0].payload["text"] == "keep"


@pytest.mark.asyncio
async def test_interrupted_collection_replacement_keeps_triplet_text_untouched(tmp_path):
    """An interrupted replacement must not clear an unrelated collection."""
    adapter = _adapter(tmp_path / "vectors")
    target_id, triplet_id = (str(uuid4()) for _ in range(2))

    await adapter.index_data_points(
        "EdgeInstance", "text", [_IndexedPoint(id=target_id, text="old")]
    )
    await adapter.index_data_points("Triplet", "text", [_IndexedPoint(id=triplet_id, text="keep")])

    with pytest.raises(RuntimeError, match="stream interrupted"):
        await adapter.replace_index_data_points(
            "EdgeInstance",
            "text",
            _interrupted_batches([_IndexedPoint(id=str(uuid4()), text="partial")]),
        )

    triplet = await adapter.retrieve("Triplet_text", [triplet_id])
    assert triplet[0].payload["text"] == "keep"
