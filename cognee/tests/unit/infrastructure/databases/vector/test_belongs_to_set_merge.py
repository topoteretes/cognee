"""Upserting the same DataPoint with different `belongs_to_set` values must
accumulate the set names, not overwrite them. This guards against losing
dataset tags when the same content is cognified into multiple datasets.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        IndexSchema,
        LanceDBAdapter,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    """Deterministic stub embedding engine; avoids external API calls in tests."""

    def get_vector_size(self):
        """Return the fixed vector dimensionality used throughout these tests."""
        return 3

    def get_batch_size(self):
        """Return the stub embedding batch size."""
        return 100

    async def embed_text(self, texts):
        """Return a fixed 3-D vector per input text."""
        return [[0.1, 0.2, 0.3] for _ in texts]


class _TaggedPoint(DataPoint):
    """Minimal DataPoint used to exercise `belongs_to_set` handling."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_belongs_to_set_merges_across_upserts(tmp_path):
    """Same id re-upserted with a new tag must keep both tags."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    collection = "Tagged_text"
    point_id = str(uuid4())

    first = _TaggedPoint(id=point_id, text="shared", belongs_to_set=["DatasetA"])
    await adapter.create_collection(collection, type(first))
    await adapter.create_data_points(collection, [first])

    second = _TaggedPoint(id=point_id, text="shared", belongs_to_set=["DatasetB"])
    await adapter.create_data_points(collection, [second])

    results = await adapter.retrieve(collection, [point_id])
    assert len(results) == 1
    assert sorted(results[0].payload["belongs_to_set"]) == ["DatasetA", "DatasetB"]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_belongs_to_set_dedupes_on_repeat_upsert(tmp_path):
    """Upserting the same payload twice should not duplicate tag entries."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    collection = "Tagged_text"
    point_id = str(uuid4())

    point = _TaggedPoint(id=point_id, text="shared", belongs_to_set=["DatasetA"])
    await adapter.create_collection(collection, type(point))
    await adapter.create_data_points(collection, [point])
    await adapter.create_data_points(collection, [point])

    results = await adapter.retrieve(collection, [point_id])
    assert results[0].payload["belongs_to_set"] == ["DatasetA"]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_belongs_to_set_first_insert_has_no_prior_tags(tmp_path):
    """A brand-new id must store exactly the tags passed in."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    collection = "Index_text"
    point_id = str(uuid4())

    await adapter.create_collection(collection, IndexSchema)
    await adapter.create_data_points(
        collection,
        [IndexSchema(id=point_id, text="fresh", belongs_to_set=["OnlyDataset"])],
    )

    results = await adapter.retrieve(collection, [point_id])
    assert results[0].payload["belongs_to_set"] == ["OnlyDataset"]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_remove_belongs_to_set_tags_strips_and_deletes(tmp_path):
    """Detag removes the tag from mixed rows, drops orphaned rows, leaves others alone."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    collection = "Tagged_text"

    shared_id = str(uuid4())
    orphaned_id = str(uuid4())
    untouched_id = str(uuid4())

    shared = _TaggedPoint(id=shared_id, text="shared", belongs_to_set=["Dev", "DevMirror"])
    orphaned = _TaggedPoint(id=orphaned_id, text="orphaned", belongs_to_set=["Dev"])
    untouched = _TaggedPoint(id=untouched_id, text="untouched", belongs_to_set=["Production"])

    await adapter.create_collection(collection, type(shared))
    await adapter.create_data_points(collection, [shared, orphaned, untouched])

    await adapter.remove_belongs_to_set_tags(["Dev"])

    surviving = await adapter.retrieve(collection, [shared_id, untouched_id])
    surviving_by_id = {str(r.id): r for r in surviving}

    assert sorted(surviving_by_id[shared_id].payload["belongs_to_set"]) == ["DevMirror"]
    assert surviving_by_id[untouched_id].payload["belongs_to_set"] == ["Production"]
    assert await adapter.retrieve(collection, [orphaned_id]) == []


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_remove_belongs_to_set_tags_noop_for_empty_input(tmp_path):
    """Calling detag with no tags must be a no-op."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    collection = "Tagged_text"
    point_id = str(uuid4())

    point = _TaggedPoint(id=point_id, text="shared", belongs_to_set=["Dev"])
    await adapter.create_collection(collection, type(point))
    await adapter.create_data_points(collection, [point])

    await adapter.remove_belongs_to_set_tags([])

    result = (await adapter.retrieve(collection, [point_id]))[0]
    assert result.payload["belongs_to_set"] == ["Dev"]
