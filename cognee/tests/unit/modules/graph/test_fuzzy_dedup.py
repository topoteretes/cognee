"""Unit tests for resolve_fuzzy_duplicate_entities (issue #3628, Approach B).

All data is synthetic and the embedding engine is faked: ``embed_data`` returns
hand-picked vectors so cosine similarity is fully under test control. No real API
is called, so the suite is deterministic in CI (the mocked-embedding harness).
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import Entity
from cognee.modules.graph.utils import resolve_fuzzy_duplicate_entities
from cognee.modules.graph.utils.fuzzy_dedup import MERGED_INTO_RELATIONSHIP


# --- synthetic embedding space --------------------------------------------
# Unit vectors on a 2D arc, chosen so every cosine similarity is exact and
# obvious. Angles from [1,0,0]: OpenAI 0deg, OpenAI Inc. ~16.26deg, Open AI
# ~32.52deg. So:
#   cos(OpenAI, OpenAI Inc.)  ~ 0.96   (high)
#   cos(OpenAI Inc., Open AI) ~ 0.96   (high)
#   cos(OpenAI, Open AI)      ~ 0.843  (BELOW 0.95 -> tests transitivity)
#   cos(OpenAI, Google)       = 0.0    (orthogonal)
# A name not listed falls back to a zero vector (cosine 0 with everything).
_VECTORS = {
    "OpenAI": [1.0, 0.0, 0.0],
    "OpenAI Inc.": [0.96, 0.28, 0.0],  # unit: 0.96^2 + 0.28^2 == 1
    "Open AI": [0.8431, 0.5378, 0.0],
    "Google": [0.0, 1.0, 0.0],
    "Microsoft": [0.0, 0.0, 1.0],
}


def _vector_for(name: str) -> list[float]:
    return _VECTORS.get(name, [0.0, 0.0, 0.0])


def _fake_vector_engine():
    engine = MagicMock()
    engine.embed_data = AsyncMock(side_effect=lambda names: [_vector_for(n) for n in names])
    return engine


def _entity(name: str, description: str = "") -> Entity:
    return Entity(name=name, description=description or f"desc of {name}")


# --- in-batch clustering ---------------------------------------------------


@pytest.mark.asyncio
async def test_high_similarity_pair_links_merged_into():
    canonical = _entity("OpenAI")
    dup = _entity("OpenAI Inc.")
    other = _entity("Google")
    nodes = [canonical, dup, other]

    edges = await resolve_fuzzy_duplicate_entities(
        nodes, _fake_vector_engine(), similarity_threshold=0.95
    )

    # Exactly one merged_into edge: the duplicate -> the first-appearing canonical.
    assert len(edges) == 1
    source, target, relationship, properties = edges[0]
    assert source == dup.id
    assert target == canonical.id
    assert relationship == MERGED_INTO_RELATIONSHIP
    assert properties["resolution"] == "embedding_similarity"
    assert properties["similarity_score"] >= 0.95

    # Non-destructive: no node ids rewritten, every node still present.
    assert {dup.id, canonical.id, other.id} == {n.id for n in nodes}


@pytest.mark.asyncio
async def test_low_similarity_pair_not_linked():
    a = _entity("OpenAI")
    b = _entity("Google")

    edges = await resolve_fuzzy_duplicate_entities(
        [a, b], _fake_vector_engine(), similarity_threshold=0.95
    )

    assert edges == []


@pytest.mark.asyncio
async def test_union_find_transitive_clustering():
    # A~B high, B~C high, but A~C is below threshold. Union-Find still groups
    # all three, and every non-canonical member links to the first-appearing
    # node (A).
    a = _entity("OpenAI")  # [1, 0, 0]
    b = _entity("OpenAI Inc.")  # ~ a and ~ c
    c = _entity("Open AI")  # ~ b, below threshold vs a directly

    edges = await resolve_fuzzy_duplicate_entities(
        [a, b, c], _fake_vector_engine(), similarity_threshold=0.95
    )

    # Two merged_into edges, both pointing at the canonical (A).
    assert len(edges) == 2
    assert all(target == a.id for _source, target, _rel, _props in edges)
    assert {source for source, _t, _r, _p in edges} == {b.id, c.id}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "threshold, should_merge",
    [(0.90, True), (0.95, True), (0.97, False)],
)
async def test_threshold_boundary(threshold, should_merge):
    canonical = _entity("OpenAI")  # [1, 0, 0]
    dup = _entity("OpenAI Inc.")  # cos ~ 0.96

    edges = await resolve_fuzzy_duplicate_entities(
        [canonical, dup], _fake_vector_engine(), similarity_threshold=threshold
    )

    assert bool(edges) is should_merge


# --- blocking / non-entity handling ---------------------------------------


class _NonEntity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


@pytest.mark.asyncio
async def test_non_entity_nodes_are_skipped():
    # A same-named non-Entity node must not participate in fuzzy clustering, and
    # only entity names get embedded (blocking narrows the candidate set).
    chunk = _NonEntity(name="OpenAI")
    a = _entity("OpenAI")
    b = _entity("OpenAI Inc.")
    engine = _fake_vector_engine()

    edges = await resolve_fuzzy_duplicate_entities([chunk, a, b], engine, similarity_threshold=0.95)

    # The two entities merge; the same-named chunk is untouched and never embedded.
    assert len(edges) == 1
    assert {edges[0][0], edges[0][1]} == {a.id, b.id}
    engine.embed_data.assert_awaited_once_with(["OpenAI", "OpenAI Inc."])


@pytest.mark.asyncio
async def test_fewer_than_two_entities_returns_empty():
    chunk = _NonEntity(name="whatever")
    lone = _entity("OpenAI")
    engine = _fake_vector_engine()

    edges = await resolve_fuzzy_duplicate_entities([chunk, lone], engine, similarity_threshold=0.95)

    assert edges == []
    engine.embed_data.assert_not_called()  # nothing to compare, no embedding call


@pytest.mark.asyncio
async def test_inputs_are_not_mutated():
    canonical = _entity("OpenAI")
    dup = _entity("OpenAI Inc.")
    nodes = [canonical, dup]
    ids_before = [n.id for n in nodes]

    await resolve_fuzzy_duplicate_entities(nodes, _fake_vector_engine(), similarity_threshold=0.95)

    assert nodes == [canonical, dup]  # list untouched
    assert [n.id for n in nodes] == ids_before  # ids untouched (non-destructive)


# --- shared-fixture precision/recall (deterministic lexical proxy) ----------

_FIXTURE = (
    Path(__file__).resolve().parents[5]
    / "examples/pocs/post_extraction_canonicalization/data/example2"
    / "expected_disambiguation_entities.txt"
)


def _bigram_vectors(names: list[str]) -> list[list[float]]:
    """Deterministic char-bigram frequency vectors — a reproducible *lexical*
    proxy for a real embedding engine, so precision/recall is stable in CI with
    no API call. The calibrated threshold below is proxy-specific, not the
    production 0.72; this test guards the clustering logic and the precision/
    recall harness, not the absolute production numbers."""

    def bigrams(s: str) -> list[str]:
        s = s.lower()
        return [s[i : i + 2] for i in range(len(s) - 1)]

    vocab = sorted({b for n in names for b in bigrams(n)})
    index = {b: i for i, b in enumerate(vocab)}
    vectors = []
    for name in names:
        vector = [0.0] * len(vocab)
        for b in bigrams(name):
            vector[index[b]] += 1.0
        vectors.append(vector)
    return vectors


def _lexical_engine():
    engine = MagicMock()
    engine.embed_data = AsyncMock(side_effect=lambda names: _bigram_vectors(names))
    return engine


def _clusters_from_merge_edges(entities, merge_edges):
    """Reconstruct which entity ids ended up merged, from the merged_into edges:
    the connected components of the (duplicate -> canonical) links."""
    parent = {str(e.id): str(e.id) for e in entities}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for source, target, *_ in merge_edges:
        parent[find(str(source))] = find(str(target))
    return {str(e.id): find(str(e.id)) for e in entities}


def _pair_precision_recall(entities, merge_edges, gt_labels):
    """Precision/recall of merge decisions: a 'merge' is a pair of entities that
    ended up in the same merged_into component, scored against ground truth."""
    component = _clusters_from_merge_edges(entities, merge_edges)
    ids = [str(e.id) for e in entities]
    tp = fp = fn = 0
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            predicted_same = component[ids[i]] == component[ids[j]]
            truth_same = gt_labels[i] == gt_labels[j]
            if predicted_same and truth_same:
                tp += 1
            elif predicted_same and not truth_same:
                fp += 1
            elif not predicted_same and truth_same:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall


@pytest.mark.asyncio
@pytest.mark.skipif(not _FIXTURE.exists(), reason="shared fixture not found")
async def test_precision_recall_on_shared_fixture():
    # 12 known "OpenAI" variants (all one ground-truth entity) + 3 distractors.
    variants = [line.strip() for line in _FIXTURE.read_text().splitlines() if line.strip()]
    distractors = ["Google", "Microsoft", "Tesla"]
    names = variants + distractors
    gt_labels = ["openai"] * len(variants) + distractors

    entities = [_entity(n) for n in names]
    # Threshold calibrated to the lexical proxy, NOT the production 0.72.
    merge_edges = await resolve_fuzzy_duplicate_entities(
        entities, _lexical_engine(), similarity_threshold=0.6
    )

    precision, recall = _pair_precision_recall(entities, merge_edges, gt_labels)

    # Conservative by design: zero false merges (distractors never merge in).
    assert precision == 1.0
    # Partial recall: catches close lexical variants, misses distant ones
    # ("the ai lab", "the a.i. lab", ...) — the documented threshold tradeoff.
    assert 0.5 < recall < 1.0

    # The obvious variants collapse into one component.
    component = _clusters_from_merge_edges(entities, merge_edges)
    by_component: dict[str, set[str]] = {}
    for e in entities:
        by_component.setdefault(component[str(e.id)], set()).add(e.name.lower())
    openai_component = max(by_component.values(), key=len)
    assert {"openai", "open ai", "openai hq"} <= openai_component
