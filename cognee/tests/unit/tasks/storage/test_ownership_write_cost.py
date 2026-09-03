"""Chunk ownership must not make ingestion cost scale with document length.

Stamping an artifact's FIRST owner is free — it folds into the statement that
writes the row. Every ADDITIONAL owner needs an attach call, and each of those
is a lock-serialized read-then-write query pair on the graph adapter. Grouping
those calls by individual ref key made their number linear in chunk count
(measured 73 extra calls at 40 chunks, 193 at 100), which is what the ingestion
regression actually was — not the model walk, which measures ~42 ms at 40
chunks.

Grouping by KEY SET instead makes the count scale with distinct sharing
patterns. These tests pin the shape, not a wall-clock number.
"""

import asyncio
from uuid import uuid4

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import Entity, EntityType
from cognee.tasks.storage.add_data_points import _group_by_extra_keys
from cognee.tasks.storage.chunk_ownership import collect_chunk_ownership
from cognee.tasks.summarization.models import TextSummary


def _batch(chunk_count: int, shared_entities: int = 12):
    """A cognify-shaped batch: summaries -> chunks -> entities -> types."""
    document = TextDocument(
        id=uuid4(),
        title="doc.txt",
        name="doc",
        raw_data_location="/tmp/doc.txt",
        mime_type="text/plain",
        external_metadata="{}",
    )
    types = [EntityType(id=uuid4(), name=f"Type{i}", description="t") for i in range(4)]
    pool = [
        Entity(id=uuid4(), name=f"Shared{i}", description="d", is_a=types[i % len(types)])
        for i in range(shared_entities)
    ]
    roots = []
    for index in range(chunk_count):
        entities = [pool[(index * 3 + k) % len(pool)] for k in range(3)]
        chunk = DocumentChunk(
            id=uuid4(),
            text=f"chunk {index}",
            chunk_size=10,
            chunk_index=index,
            cut_type="paragraph_end",
            is_part_of=document,
            contains=entities,
        )
        roots.append(TextSummary(id=uuid4(), text=f"summary {index}", made_from=chunk))
    return roots


def _call_counts(chunk_count: int):
    """(grouped-by-key calls, grouped-by-key-set calls) for one batch."""
    roots = _batch(chunk_count)
    ownership = asyncio.run(collect_chunk_ownership(roots, uuid4(), uuid4()))

    by_key = set()
    for owners in list(ownership.node_owners.values()) + list(ownership.edge_owners.values()):
        by_key.update(owners[1:])

    by_key_set = len(_group_by_extra_keys(ownership.node_owners)) + len(
        _group_by_extra_keys(ownership.edge_owners)
    )
    return len(by_key), by_key_set


def test_grouping_preserves_every_artifacts_key_set():
    """The optimisation must be semantically invisible.

    Every artifact still receives exactly the extra keys it had; only the
    number of calls carrying them changes.
    """
    roots = _batch(12)
    ownership = asyncio.run(collect_chunk_ownership(roots, uuid4(), uuid4()))

    for owner_map in (ownership.node_owners, ownership.edge_owners):
        delivered: dict = {}
        for keys, artifacts in _group_by_extra_keys(owner_map).items():
            for artifact in artifacts:
                delivered.setdefault(artifact, set()).update(keys)

        for artifact, owners in owner_map.items():
            assert delivered.get(artifact, set()) == set(owners[1:]), artifact


def test_first_owner_is_never_attached_separately():
    """It folds into the write statement; attaching it again would be a wasted pair."""
    roots = _batch(8)
    ownership = asyncio.run(collect_chunk_ownership(roots, uuid4(), uuid4()))

    for owner_map in (ownership.node_owners, ownership.edge_owners):
        for keys, artifacts in _group_by_extra_keys(owner_map).items():
            for artifact in artifacts:
                assert owner_map[artifact][0] not in keys


def test_artifacts_without_extra_owners_cost_nothing():
    assert _group_by_extra_keys({"solo": ["only-key"]}) == {}


def test_call_count_stops_scaling_with_chunk_count():
    """The regression itself: per-key grouping grows with the document.

    Ten times the chunks must not mean ten times the round-trips.
    """
    small_by_key, small_grouped = _call_counts(10)
    large_by_key, large_grouped = _call_counts(100)

    assert large_by_key > small_by_key * 5, (
        "fixture is not exercising growth; per-key grouping should scale with chunks"
    )
    assert large_grouped < large_by_key / 2, (
        f"grouping by key set saved too little: {large_grouped} vs {large_by_key} calls"
    )
    # The real property: the grouped count is bounded by sharing structure, so
    # it must not grow proportionally with document length the way per-key does.
    assert large_grouped < small_grouped * 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
