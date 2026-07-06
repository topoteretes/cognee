"""Unit tests for cluster_fuzzy_duplicate_entities (issue #3628, Approach B).

All data here is synthetic. The embedding engine is faked: `embed_data` returns
hand-picked vectors so cosine similarity is fully under test control, and
`batch_search` returns hand-built `ScoredResult`s. No real API is called.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.engine import DataPoint
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.engine.models import Entity
from cognee.modules.graph.utils import (
    cluster_fuzzy_duplicate_entities,
    deduplicate_nodes_and_edges,
)


# --- synthetic embedding space --------------------------------------------
# Unit vectors on a 2D arc, chosen so every cosine similarity is exact and
# obvious. Angles from [1,0,0]: OpenAI 0deg, OpenAI Inc. 16.26deg, Open AI
# 32.52deg. So:
#   cos(OpenAI, OpenAI Inc.)      ~ 0.96   (high)
#   cos(OpenAI Inc., Open AI)     ~ 0.96   (high)
#   cos(OpenAI, Open AI)          ~ 0.843  (BELOW 0.95 -> tests transitivity)
#   cos(OpenAI, Google)           = 0.0    (orthogonal)
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


def _fake_vector_engine(search_results=None, search_error=None):
    engine = MagicMock()
    engine.embed_data = AsyncMock(side_effect=lambda names: [_vector_for(n) for n in names])
    if search_error is not None:
        engine.batch_search = AsyncMock(side_effect=search_error)
    else:
        engine.batch_search = AsyncMock(return_value=search_results or [])
    return engine


def _entity(name: str, description: str = "") -> Entity:
    return Entity(name=name, description=description or f"desc of {name}")


def _edge(source, target, rel="related_to"):
    """Edge tuple mirroring production shape, with ids embedded in properties."""
    return (
        source.id,
        target.id,
        rel,
        {"source_node_id": source.id, "target_node_id": target.id, "relationship_name": rel},
    )


# --- in-batch clustering ---------------------------------------------------


@pytest.mark.asyncio
async def test_in_batch_high_similarity_pair_merges_onto_first():
    canonical = _entity("OpenAI")
    dup = _entity("OpenAI Inc.")
    other = _entity("Google")
    # Edge from a chunk-like node into the duplicate, plus its properties.
    edge = _edge(dup, other)

    nodes = [canonical, dup, other]
    engine = _fake_vector_engine()

    result_nodes, result_edges, records = await cluster_fuzzy_duplicate_entities(
        nodes, [edge], engine, similarity_threshold=0.95
    )

    # Duplicate's id was rewritten to the first-appearing canonical.
    assert dup.id == canonical.id
    assert other.id != canonical.id  # low similarity, untouched

    # Edge endpoint AND embedded properties ids were remapped.
    remapped = result_edges[0]
    assert remapped[0] == canonical.id
    assert remapped[3]["source_node_id"] == canonical.id
    assert remapped[3]["target_node_id"] == other.id

    # One in-batch merge recorded.
    assert len(records) == 1
    assert records[0]["scope"] == "in_batch"
    assert records[0]["duplicate_name"] == "OpenAI Inc."
    assert records[0]["canonical_id"] == str(canonical.id)

    # Downstream dedup (unchanged) collapses the now-identical ids to one node.
    final_nodes, _ = deduplicate_nodes_and_edges(result_nodes, result_edges)
    entity_ids = {str(n.id) for n in final_nodes if type(n).__name__ == "Entity"}
    assert str(canonical.id) in entity_ids
    assert len([n for n in final_nodes if str(n.id) == str(canonical.id)]) == 1


@pytest.mark.asyncio
async def test_low_similarity_pair_not_merged():
    a = _entity("OpenAI")
    b = _entity("Google")
    nodes = [a, b]
    engine = _fake_vector_engine()

    result_nodes, _, records = await cluster_fuzzy_duplicate_entities(
        nodes, [], engine, similarity_threshold=0.95
    )

    assert a.id != b.id
    assert records == []
    assert len(result_nodes) == 2


@pytest.mark.asyncio
async def test_merged_from_provenance_written_on_canonical():
    canonical = _entity("OpenAI")
    dup = _entity("OpenAI Inc.")
    engine = _fake_vector_engine()

    await cluster_fuzzy_duplicate_entities([canonical, dup], [], engine, similarity_threshold=0.95)

    merged_from = canonical.metadata["merged_from"]
    assert len(merged_from) == 1
    entry = merged_from[0]
    assert entry["duplicate_name"] == "OpenAI Inc."
    assert entry["method"] == "embedding_similarity"
    assert entry["similarity"] >= 0.95
    assert entry["threshold"] == 0.95


@pytest.mark.asyncio
async def test_union_find_transitive_clustering():
    # A~B high, B~C high, but A~C is below threshold. Union-Find still groups
    # all three, and everything merges onto the first-appearing node (A).
    a = _entity("OpenAI")  # [1, 0, 0]
    b = _entity("OpenAI Inc.")  # ~ a and ~ c
    c = _entity("Open AI")  # ~ b, slightly further from a
    engine = _fake_vector_engine()

    result_nodes, _, records = await cluster_fuzzy_duplicate_entities(
        [a, b, c], [], engine, similarity_threshold=0.95
    )

    assert b.id == a.id
    assert c.id == a.id
    assert {r["duplicate_name"] for r in records} == {"OpenAI Inc.", "Open AI"}

    final_nodes, _ = deduplicate_nodes_and_edges(result_nodes, [])
    assert len([n for n in final_nodes if str(n.id) == str(a.id)]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "threshold, should_merge",
    [(0.90, True), (0.95, True), (0.97, False)],
)
async def test_threshold_boundary(threshold, should_merge):
    canonical = _entity("OpenAI")  # [1, 0, 0]
    dup = _entity("OpenAI Inc.")  # cos ~ 0.96
    engine = _fake_vector_engine()

    _, _, records = await cluster_fuzzy_duplicate_entities(
        [canonical, dup], [], engine, similarity_threshold=threshold
    )

    assert bool(records) is should_merge


# --- cross-batch matching against existing graph nodes ---------------------


@pytest.mark.asyncio
async def test_cross_batch_merges_into_existing_and_removes_duplicate():
    existing_id = uuid4()
    dup = _entity("OpenAI Inc.")
    other = _entity("Google")
    edge = _edge(dup, other)

    # batch_search returns the pre-existing "OpenAI" node for the first query
    # (the dup's cluster rep) and nothing for the second (Google).
    search_results = [
        [ScoredResult(id=existing_id, score=0.02, payload={"name": "OpenAI"})],  # sim 0.98
        [],
    ]
    engine = _fake_vector_engine(search_results=search_results)

    result_nodes, result_edges, records = await cluster_fuzzy_duplicate_entities(
        [dup, other], [edge], engine, similarity_threshold=0.95
    )

    # The batch duplicate is REMOVED so it can never overwrite the existing node.
    assert dup not in result_nodes
    assert other in result_nodes

    # Its edge now points at the existing node (tuple + properties).
    remapped = result_edges[0]
    assert remapped[0] == existing_id
    assert remapped[3]["source_node_id"] == existing_id

    # Cross-batch provenance lives in the returned records (canonical isn't in
    # this batch, so there's no node to attach merged_from to).
    assert len(records) == 1
    assert records[0]["scope"] == "cross_batch"
    assert records[0]["canonical_id"] == str(existing_id)
    assert records[0]["canonical_name"] == "OpenAI"


@pytest.mark.asyncio
async def test_cross_batch_below_threshold_is_ignored():
    existing_id = uuid4()
    dup = _entity("OpenAI Inc.")
    # Existing match, but distance too large -> similarity 0.80 < 0.95.
    search_results = [[ScoredResult(id=existing_id, score=0.20, payload={"name": "Something"})]]
    engine = _fake_vector_engine(search_results=search_results)

    result_nodes, _, records = await cluster_fuzzy_duplicate_entities(
        [dup], [], engine, similarity_threshold=0.95
    )

    assert dup in result_nodes
    assert records == []


@pytest.mark.asyncio
async def test_batch_search_own_batch_id_is_not_a_cross_hit():
    # A degenerate adapter that echoes the query's own (not-yet-indexed) id must
    # not be treated as a cross-batch match.
    dup = _entity("OpenAI Inc.")
    search_results = [[ScoredResult(id=dup.id, score=0.0, payload={"name": "OpenAI Inc."})]]
    engine = _fake_vector_engine(search_results=search_results)

    result_nodes, _, records = await cluster_fuzzy_duplicate_entities(
        [dup], [], engine, similarity_threshold=0.95
    )

    assert dup in result_nodes
    assert records == []


@pytest.mark.asyncio
async def test_collection_not_found_falls_back_to_in_batch():
    # First ingest: the Entity_name collection doesn't exist yet. The function
    # must not crash and should still do in-batch clustering.
    canonical = _entity("OpenAI")
    dup = _entity("OpenAI Inc.")
    engine = _fake_vector_engine(search_error=CollectionNotFoundError("Collection not found!"))

    result_nodes, _, records = await cluster_fuzzy_duplicate_entities(
        [canonical, dup], [], engine, similarity_threshold=0.95
    )

    assert dup.id == canonical.id  # in-batch merge still happened
    assert len(records) == 1
    assert records[0]["scope"] == "in_batch"


# --- blocking / non-entity handling ---------------------------------------


class _NonEntity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


@pytest.mark.asyncio
async def test_non_entity_nodes_are_skipped():
    # A same-named non-Entity node must not participate in fuzzy clustering.
    chunk = _NonEntity(name="OpenAI")
    entity = _entity("OpenAI")
    engine = _fake_vector_engine()

    result_nodes, _, records = await cluster_fuzzy_duplicate_entities(
        [chunk, entity], [], engine, similarity_threshold=0.95
    )

    assert records == []
    assert chunk in result_nodes and entity in result_nodes
    # Only entity names get embedded (blocking narrows the candidate set).
    engine.embed_data.assert_awaited_once_with(["OpenAI"])


@pytest.mark.asyncio
async def test_no_entities_returns_inputs_unchanged():
    chunk = _NonEntity(name="whatever")
    engine = _fake_vector_engine()

    result_nodes, result_edges, records = await cluster_fuzzy_duplicate_entities(
        [chunk], [], engine, similarity_threshold=0.95
    )

    assert result_nodes == [chunk]
    assert records == []
    engine.embed_data.assert_not_called()


# --- shared-fixture precision/recall (deterministic lexical proxy) ----------

_FIXTURE = (
    Path(__file__).resolve().parents[5]
    / "examples/pocs/post_extraction_canonicalization/data/example2"
    / "expected_disambiguation_entities.txt"
)


def _bigram_vectors(names: list[str]) -> list[list[float]]:
    """Deterministic char-bigram frequency vectors.

    A reproducible *lexical* proxy for a real embedding engine, so precision/
    recall is stable in CI with no API call. Absolute numbers differ from
    production semantic embeddings (bigram overlap != semantic distance, and
    the calibrated threshold below is proxy-specific, not the production 0.95);
    this test guards the clustering logic and the precision/recall harness, and
    can be re-pointed at real embeddings offline for production numbers.
    """

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
    engine.batch_search = AsyncMock(return_value=[])  # in-batch clustering only
    return engine


def _pair_precision_recall(entities, gt_labels):
    """Precision/recall of merge decisions.

    A 'merge' is a pair of entities that ended up sharing a canonical id,
    scored against ground-truth cluster labels.
    """
    ids = [str(e.id) for e in entities]
    tp = fp = fn = 0
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            predicted_same = ids[i] == ids[j]
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
    # Threshold calibrated to the lexical proxy, NOT the production 0.95.
    await cluster_fuzzy_duplicate_entities(
        entities, [], _lexical_engine(), similarity_threshold=0.6
    )

    precision, recall = _pair_precision_recall(entities, gt_labels)

    # Conservative by design: zero false merges (distractors never merge in).
    assert precision == 1.0
    # Partial recall: catches close lexical variants, misses distant ones
    # ("the ai lab", "the a.i. lab", ...) — the documented threshold tradeoff.
    assert 0.5 < recall < 1.0

    # The obvious variants collapse into one cluster.
    by_id: dict[str, set[str]] = {}
    for e in entities:
        by_id.setdefault(str(e.id), set()).add(e.name.lower())
    openai_cluster = max(by_id.values(), key=len)
    assert {"openai", "open ai", "openai hq"} <= openai_cluster
    # Recall < 1: the 12 true duplicates do NOT all collapse into one node —
    # lexically-distant variants stay separate (the documented tradeoff).
    variant_ids = {str(entities[i].id) for i in range(len(variants))}
    assert len(variant_ids) > 1


@pytest.mark.asyncio
@pytest.mark.skipif(not _FIXTURE.exists(), reason="shared fixture not found")
@pytest.mark.skipif(
    not (os.getenv("LLM_API_KEY") or os.getenv("EMBEDDING_API_KEY")),
    reason="requires a real LLM_API_KEY/EMBEDDING_API_KEY — opt-in, not run in CI",
)
async def test_precision_recall_with_real_embeddings():
    """One-time report: real embedding-provider precision/recall on the shared
    fixture, at the shipped production threshold (config default). Skipped
    unless a real API key is set locally (see the PR doc for how to set
    LLM_API_KEY without pasting it anywhere). Not part of CI — the
    deterministic lexical-proxy test above is what CI runs.

    Threshold history: calibrated against text-embedding-3-large on this same
    fixture — precision=1.0/recall=1.0 held for 0.50-0.60, recall collapsed to
    0.682 at 0.65, 0.439 at 0.70-0.72, and 0.242 at 0.80; precision started
    degrading below 0.45 (unrelated companies merging). Shipped default is
    0.72, not the precision/recall-optimal 0.50-0.60 band: at low thresholds,
    Union-Find transitivity chains merges through intermediate names (observed
    root<->member direct similarity as low as ~0.50), which raises the risk of
    an unrelated "bridge" name over-merging two different entities on a larger,
    noisier real dataset. 0.72 trades recall for shorter/rarer chains — though
    even at 0.72 a single-hop bridge was observed ("open ai" merged via
    "openai", 0.7283, despite a direct similarity to the cluster root of only
    0.6167), so transitivity risk is reduced, not eliminated, by the threshold
    alone. Known model-specific weak spot: abbreviation-expansion variants
    (e.g. "NASA" vs "The National Aeronautics and Space Administration") score
    systematically lower (~0.55-0.68) than spelling variants and are missed
    entirely at this threshold. See cognee/modules/cognify/config.py for the
    full rationale.

    Run with `-s` to see the printed numbers:
        pytest .../test_cluster_fuzzy_duplicate_entities.py::test_precision_recall_with_real_embeddings -s
    """
    variants = [line.strip() for line in _FIXTURE.read_text().splitlines() if line.strip()]
    distractors = ["Google", "Microsoft", "Tesla"]
    names = variants + distractors
    gt_labels = ["openai"] * len(variants) + distractors

    entities = [_entity(n) for n in names]
    vector_engine = await get_vector_engine()  # real embedding provider, from .env
    threshold = get_cognify_config().fuzzy_entity_dedup_threshold
    await cluster_fuzzy_duplicate_entities(
        entities, [], vector_engine, similarity_threshold=threshold
    )

    precision, recall = _pair_precision_recall(entities, gt_labels)

    # NOTE: one-time report for the issue #3628 comparison table. Comment this
    # print out (or delete it) once the numbers are recorded there — it's not
    # needed for the test to pass, only to surface the numbers with `-s`.
    print(
        f"\n[Approach B / real embeddings] threshold={threshold:.2f} "
        f"precision={precision:.3f} recall={recall:.3f}"
    )

    # Regression guard at the shipped default, measured on this fixture.
    # Recall is intentionally low here (precision/safety over coverage — see
    # docstring); banded rather than exact to tolerate minor embedding drift.
    assert precision == 1.0
    assert 0.30 < recall < 0.55


# --- property conflict resolution (strict reading: show conflict handling) --


@pytest.mark.asyncio
async def test_property_conflict_resolution_first_appearing_wins():
    # Two mentions of the same entity carry conflicting descriptions — a
    # contradiction surfaced at merge time. Policy: first-appearing wins; the
    # losing value is recorded in provenance for audit / reversal.
    canonical = _entity("OpenAI", description="AI research company, SF")
    dup = _entity("OpenAI Inc.", description="Delaware C-corp, founded 2015")

    engine = _fake_vector_engine()  # controlled vectors: cos ~0.96, they merge
    result_nodes, _, records = await cluster_fuzzy_duplicate_entities(
        [canonical, dup], [], engine, similarity_threshold=0.95
    )

    # Winner keeps its own description; loser's is discarded, not blended.
    survivors = [n for n in result_nodes if str(n.id) == str(canonical.id)]
    assert survivors and survivors[0].description == "AI research company, SF"
    # The discarded conflicting value is auditable in the merge record...
    assert records[0]["duplicate_description"] == "Delaware C-corp, founded 2015"
    # ...and persisted on the canonical node for audit / reversal.
    assert (
        canonical.metadata["merged_from"][0]["duplicate_description"]
        == "Delaware C-corp, founded 2015"
    )
