"""Unit tests for the gather + blocking helpers of the canonicalization task
(issue #3629, commit 2). The judge/merge/audit logic is not implemented yet, so
only ``_gather_entities``, ``_block_candidate_pairs``, and ``_dedup_relations``
are exercised here. All external calls (the embedding engine) are mocked; no real
API keys are required."""

import logging
import sys
from types import SimpleNamespace

import pytest
from unittest.mock import patch

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.cognify.config import CognifyConfig
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document
from cognee.tasks.summarization.models import TextSummary
from cognee.tasks.temporal_graph.models import Event
from cognee.tasks.graph.canonicalize_entities import (
    _gather_entities,
    _block_candidate_pairs,
    _dedup_relations,
)

ce = sys.modules["cognee.tasks.graph.canonicalize_entities"]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
class _FakeEmbedder:
    """Deterministic per-text embedder.

    MockEmbeddingEngine returns a *constant* vector for every input, which would
    make cosine blocking meaningless (everything looks identical). This fake maps
    each name to a fixed vector so near-duplicates (Alice/Alicia/Alyce) are highly
    similar and unrelated names (Acme) are orthogonal.
    """

    _VECTORS = {
        "alice": [1.0, 0.0, 0.0],
        "alicia": [0.98, 0.03, 0.0],
        "alyce": [0.97, 0.05, 0.0],
        "acme": [0.0, 1.0, 0.0],
    }

    async def embed_text(self, texts):
        return [self._VECTORS.get(text.lower(), [0.0, 0.0, 1.0]) for text in texts]


def _patch_vector_engine():
    """Patch get_vector_engine in the task module to use the deterministic fake."""
    return patch.object(
        ce,
        "get_vector_engine",
        return_value=SimpleNamespace(embedding_engine=_FakeEmbedder()),
    )


def _entity(name, description="desc"):
    return Entity(name=name, description=description)


def _make_summary(contains):
    """Build a real TextSummary whose made_from chunk holds the given contains list.

    contains is assigned post-construction so mixed/edge-case items don't trip
    pydantic validation (DataPoint does not validate on assignment)."""
    document = Document(
        name="Doc",
        raw_data_location="memory",
        external_metadata=None,
        mime_type="text/plain",
    )
    chunk = DocumentChunk(
        text="chunk text",
        chunk_size=10,
        chunk_index=0,
        cut_type="paragraph",
        is_part_of=document,
    )
    chunk.contains = contains
    return TextSummary(text="summary", made_from=chunk)


# ---------------------------------------------------------------------------
# _gather_entities
# ---------------------------------------------------------------------------
def test_gather_entities_handles_tuple_bare_and_skips_others():
    alice = _entity("Alice")
    bob = _entity("Bob")
    person_type = EntityType(name="Person", description="A person type")

    summary = _make_summary(
        [
            (Edge(relationship_type="contains"), alice),  # (Edge, Entity) -> kept
            bob,  # bare Entity -> kept
            Event(name="Some meeting"),  # Event -> skipped
            (Edge(relationship_type="contains"), person_type),  # (Edge, non-Entity) -> skipped
            object(),  # bare non-Entity -> skipped
        ]
    )

    result = _gather_entities([summary])
    names = {e.name for e in result}
    assert names == {"Alice", "Bob"}


def test_gather_entities_skips_non_textsummary_rows():
    alice = _entity("Alice")
    summary = _make_summary([(Edge(relationship_type="contains"), alice)])

    # A raw (non-TextSummary) row — e.g. a DLT row passed through by summarize_text.
    dlt_row = object()

    result = _gather_entities([summary, dlt_row])
    assert [e.name for e in result] == ["Alice"]


def test_gather_entities_dedups_by_id_first_wins():
    # Two Entity objects with the same name resolve to the same id (identity_fields
    # = ["name"]); the first occurrence must win.
    first = _entity("Alice", description="first")
    second = _entity("Alice", description="second")
    assert str(first.id) == str(second.id)

    summary = _make_summary([first, second])
    result = _gather_entities([summary])
    assert len(result) == 1
    assert result[0].description == "first"


def test_gather_entities_empty_and_missing_contains():
    empty_summary = _make_summary([])
    none_summary = _make_summary(None)
    assert _gather_entities([empty_summary, none_summary]) == []


# ---------------------------------------------------------------------------
# _block_candidate_pairs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_block_candidate_pairs_returns_only_near_dupes():
    alice = _entity("Alice")
    alicia = _entity("Alicia")
    acme = _entity("Acme")
    cfg = CognifyConfig()  # similarity threshold 0.8

    with _patch_vector_engine():
        pairs = await _block_candidate_pairs([alice, alicia, acme], cfg)

    assert len(pairs) == 1
    pair_names = {pairs[0][0].name, pairs[0][1].name}
    assert pair_names == {"Alice", "Alicia"}
    # Acme is orthogonal to the Alice cluster and must not appear in any pair.
    assert all("Acme" not in {a.name, b.name} for a, b in pairs)


@pytest.mark.asyncio
async def test_block_candidate_pairs_caps_and_logs(caplog):
    # Three mutually near-identical entities -> 3 candidate pairs; cap to 1.
    entities = [_entity("Alice"), _entity("Alicia"), _entity("Alyce")]
    cfg = CognifyConfig(canonicalization_max_pairs=1)

    with _patch_vector_engine():
        with caplog.at_level(logging.INFO):
            pairs = await _block_candidate_pairs(entities, cfg)

    assert len(pairs) == 1
    assert any("capping candidate pairs" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_block_candidate_pairs_no_pairs_below_threshold():
    # Two orthogonal entities -> no pair clears the threshold.
    acme = _entity("Acme")
    other = _entity("Zeta")  # maps to default [0,0,1], orthogonal to Acme's [0,1,0]
    cfg = CognifyConfig()

    with _patch_vector_engine():
        pairs = await _block_candidate_pairs([acme, other], cfg)

    assert pairs == []


# ---------------------------------------------------------------------------
# _dedup_relations
# ---------------------------------------------------------------------------
def test_dedup_relations_keeps_first_by_type_and_target():
    target_a = _entity("Alice")
    target_b = _entity("Bob")

    relations = [
        (Edge(relationship_type="knows"), target_a),
        (Edge(relationship_type="knows"), target_a),  # duplicate -> dropped
        (Edge(relationship_type="likes"), target_a),  # diff type -> kept
        (Edge(relationship_type="knows"), target_b),  # diff target -> kept
    ]

    result = _dedup_relations(relations)
    assert len(result) == 3
    keys = {(edge.relationship_type, target.name) for edge, target in result}
    assert keys == {("knows", "Alice"), ("likes", "Alice"), ("knows", "Bob")}
