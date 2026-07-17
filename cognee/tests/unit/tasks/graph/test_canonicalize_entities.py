"""Unit tests for the canonicalization task (issue #3629): entity gathering,
blocking, and the LLM-judge + rename/mirror merge gate. All external calls (the
embedding engine and the LLM) are mocked; no real API keys are required. The
merge tests drive the REAL get_graph_from_model + deduplicate_nodes_and_edges to
prove edges survive a merge (the trap-2 guarantee)."""

import logging
import sys
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.cognify.config import CognifyConfig
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model
from cognee.modules.graph.utils.deduplicate_nodes_and_edges import (
    deduplicate_nodes_and_edges,
)
from cognee.tasks.summarization.models import TextSummary
from cognee.tasks.temporal_graph.models import Event
from cognee.tasks.graph.models import CanonicalizationJudgment, PairJudgment
from cognee.tasks.graph.canonicalize_entities import (
    _gather_entities,
    _block_candidate_pairs,
    _dedup_relations,
    _select_winner,
    _merge_component,
    _apply_merges,
    canonicalize_entities,
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


# ---------------------------------------------------------------------------
# Merge-gate helpers
# ---------------------------------------------------------------------------
def _entity_with_relation(name, rel_type, target, description="desc"):
    entity = _entity(name, description)
    entity.relations = [(Edge(relationship_type=rel_type), target)]
    return entity


def _judgment(pair_index, is_same, canonical, reconciled, confidence, rationale="because"):
    return PairJudgment(
        pair_index=pair_index,
        is_same_entity=is_same,
        canonical_name=canonical,
        reconciled_description=reconciled,
        confidence=confidence,
        rationale=rationale,
    )


def _patch_llm(judgments):
    """Patch the LLM judge seam to return a canned CanonicalizationJudgment."""
    mock = AsyncMock(return_value=CanonicalizationJudgment(judgments=judgments))
    return patch.object(ce.LLMGateway, "acreate_structured_output", mock)


async def _flatten(summaries):
    """Run the REAL get_graph_from_model + deduplicate_nodes_and_edges over a
    summary list, exactly as add_data_points does (shared visited dicts)."""
    added_nodes, added_edges, visited = {}, {}, {}
    all_nodes, all_edges = [], []
    for summary in summaries:
        nodes, edges = await get_graph_from_model(
            summary,
            added_nodes=added_nodes,
            added_edges=added_edges,
            visited_properties=visited,
        )
        all_nodes.extend(nodes)
        all_edges.extend(edges)
    return deduplicate_nodes_and_edges(all_nodes, all_edges)


# ---------------------------------------------------------------------------
# (b) THE CRITICAL CASE: merge collapses to one node AND edges survive
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_merge_collapses_and_both_edges_survive():
    acme = _entity("Acme", "a company")
    globex = _entity("Globex", "another company")
    # Two near-dup entities, EACH with a DISTINCT outbound relation.
    alice = _entity_with_relation("Alice", "works_at", acme, "engineer")
    alicia = _entity_with_relation("Alicia", "founded", globex, "software engineer")

    summary = _make_summary(
        [
            (Edge(relationship_type="contains"), alice),
            (Edge(relationship_type="contains"), alicia),
        ]
    )

    judgments = [_judgment(0, True, "Alice", "Engineer who founded Globex.", 0.95)]

    with _patch_vector_engine(), _patch_llm(judgments):
        await canonicalize_entities([summary])

    nodes, edges = await _flatten([summary])

    canonical_id = str(Entity.id_for("Alice"))
    survivors = [n for n in nodes if str(n.id) == canonical_id]
    assert len(survivors) == 1  # trap-1: dedup collapsed the duplicate
    assert survivors[0].description == "Engineer who founded Globex."

    # trap-2: BOTH the winner's and the loser's outbound edges survive the merge.
    targets_from_canonical = {str(e[1]) for e in edges if str(e[0]) == canonical_id}
    assert str(acme.id) in targets_from_canonical
    assert str(globex.id) in targets_from_canonical


# ---------------------------------------------------------------------------
# (c) no-merge -> both entities survive unchanged
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_merge_keeps_both_entities():
    alice = _entity("Alice", "engineer")
    alicia = _entity("Alicia", "sw engineer")
    original_alice_id, original_alicia_id = str(alice.id), str(alicia.id)

    summary = _make_summary(
        [
            (Edge(relationship_type="contains"), alice),
            (Edge(relationship_type="contains"), alicia),
        ]
    )

    judgments = [_judgment(0, False, "Alice", "unused", 0.99)]

    with _patch_vector_engine(), _patch_llm(judgments):
        await canonicalize_entities([summary])

    assert str(alice.id) == original_alice_id
    assert str(alicia.id) == original_alicia_id
    assert alice.id != alicia.id
    assert not getattr(alice, "merged_aliases", None)


# ---------------------------------------------------------------------------
# (d) confidence gate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "confidence, should_merge",
    [(0.70, False), (0.90, True)],  # default threshold is 0.85
)
async def test_confidence_gate(confidence, should_merge):
    alice = _entity("Alice", "engineer")
    alicia = _entity("Alicia", "sw engineer")

    summary = _make_summary(
        [
            (Edge(relationship_type="contains"), alice),
            (Edge(relationship_type="contains"), alicia),
        ]
    )

    judgments = [_judgment(0, True, "Alice", "merged", confidence)]

    with _patch_vector_engine(), _patch_llm(judgments):
        await canonicalize_entities([summary])

    if should_merge:
        assert str(alicia.id) == str(Entity.id_for("Alice"))
        assert str(alice.id) == str(alicia.id)
    else:
        assert str(alice.id) != str(alicia.id)


# ---------------------------------------------------------------------------
# (e) audit / provenance stamped on survivor + exactly one structured log
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_merge_stamps_audit_metadata_and_logs(caplog):
    alice = _entity("Alice", "engineer")
    alicia = _entity("Alicia", "sw engineer")

    summary = _make_summary(
        [
            (Edge(relationship_type="contains"), alice),
            (Edge(relationship_type="contains"), alicia),
        ]
    )

    judgments = [_judgment(0, True, "Alice", "merged desc", 0.93)]
    ctx = SimpleNamespace(pipeline_run_id="run-1", dataset=SimpleNamespace(id="ds-1"))

    with _patch_vector_engine(), _patch_llm(judgments):
        with caplog.at_level(logging.INFO):
            await canonicalize_entities([summary], ctx=ctx)

    # alice is the winner (canonical "Alice" matches its normalized name).
    assert getattr(alice, "merge_confidence", None) == 0.93
    assert "Alicia" in (getattr(alice, "merged_aliases", None) or [])
    assert alice.source_task == "canonicalize_entities"
    assert alice.version == 2  # bumped once from the default version 1

    merge_logs = [r for r in caplog.records if "entity_canonicalization_merge" in r.getMessage()]
    assert len(merge_logs) == 1


# ---------------------------------------------------------------------------
# union-find: transitive chain A~B, B~C collapses to one id
# ---------------------------------------------------------------------------
def test_apply_merges_union_find_transitive_chain():
    alpha = _entity("Alpha", "one")
    beta = _entity("Beta", "two")
    gamma = _entity("Gamma", "three")

    judgments = [
        (_judgment(0, True, "Alpha", "merged", 0.95), alpha, beta),
        (_judgment(1, True, "Alpha", "merged", 0.95), beta, gamma),
    ]

    _apply_merges([alpha, beta, gamma], judgments, CognifyConfig(), ctx=None)

    canonical_id = str(Entity.id_for("Alpha"))
    assert str(alpha.id) == canonical_id
    assert str(beta.id) == canonical_id
    assert str(gamma.id) == canonical_id


# ---------------------------------------------------------------------------
# _merge_component primitive + order-independence regression tests
#
# get_graph_from_model materializes the surviving node from whichever component
# member it reaches FIRST (extraction/contains order, independent of winner
# selection). These pin that the collapse is lossless regardless of that order:
# every member ends up identical, so no edge / provenance / type is dropped when
# a loser is traversed first.
# ---------------------------------------------------------------------------
def test_merge_component_makes_all_members_identical():
    acme = _entity("Acme")
    globex = _entity("Globex")
    winner = _entity_with_relation("Alice", "works_at", acme)
    loser = _entity_with_relation("Alicia", "founded", globex)

    _merge_component(winner, [loser], _judgment(0, True, "Alice", "merged", 0.9), ctx=None)

    canonical_id = str(Entity.id_for("Alice"))  # the winner's own id
    assert str(winner.id) == canonical_id
    assert str(loser.id) == canonical_id  # loser collapsed onto the winner's id
    assert winner.description == loser.description == "merged"
    # union of both members' outbound relations, carried by BOTH objects
    winner_targets = {t.name for _e, t in winner.relations}
    loser_targets = {t.name for _e, t in loser.relations}
    assert winner_targets == {"Acme", "Globex"}
    assert loser_targets == {"Acme", "Globex"}
    # provenance is mirrored onto the loser too, so it survives whichever member
    # get_graph_from_model materializes into the canonical node.
    assert loser.merged_aliases == winner.merged_aliases == ["Alicia"]
    assert loser.merge_confidence == winner.merge_confidence == 0.9
    assert loser.version == winner.version


@pytest.mark.asyncio
async def test_transitive_merge_preserves_every_members_edges_loser_first():
    # 3 near-dup entities, each with a DISTINCT outbound edge, merged transitively.
    # A LOSER is listed first in `contains`, so it is the object that materializes
    # into the canonical node — the pre-fix stale-relations bug dropped the
    # last-merged member's edge here.
    company_a, company_b, company_c = _entity("CompanyA"), _entity("CompanyB"), _entity("CompanyC")
    alice = _entity_with_relation("Alice", "works_at", company_a, "a")
    alicia = _entity_with_relation("Alicia", "works_at", company_b, "b")
    alyce = _entity_with_relation("Alyce", "works_at", company_c, "c")

    summary = _make_summary(
        [
            (Edge(relationship_type="contains"), alicia),  # loser first
            (Edge(relationship_type="contains"), alyce),  # loser second
            (Edge(relationship_type="contains"), alice),  # winner last
        ]
    )
    # Two confirmed pairs over three nodes → one transitive component of all three.
    judgments = [
        _judgment(0, True, "Alice", "merged", 0.95),
        _judgment(1, True, "Alice", "merged", 0.95),
    ]

    with _patch_vector_engine(), _patch_llm(judgments):
        await canonicalize_entities([summary])

    nodes, edges = await _flatten([summary])
    canonical_id = str(Entity.id_for("Alice"))
    assert len([n for n in nodes if str(n.id) == canonical_id]) == 1
    targets = {str(e[1]) for e in edges if str(e[0]) == canonical_id and e[2] == "works_at"}
    # EVERY member's distinct outbound edge survives, regardless of traversal order.
    assert {str(company_a.id), str(company_b.id), str(company_c.id)} <= targets


@pytest.mark.asyncio
async def test_merge_provenance_persists_on_node_when_loser_traversed_first():
    alice = _entity("Alice", "engineer")
    alicia = _entity("Alicia", "sw engineer")
    # Loser (Alicia) listed first → it materializes into the canonical node.
    summary = _make_summary(
        [
            (Edge(relationship_type="contains"), alicia),
            (Edge(relationship_type="contains"), alice),
        ]
    )
    judgments = [_judgment(0, True, "Alice", "merged desc", 0.93)]

    with _patch_vector_engine(), _patch_llm(judgments):
        await canonicalize_entities([summary])

    nodes, _edges = await _flatten([summary])
    canonical_id = str(Entity.id_for("Alice"))
    survivors = [n for n in nodes if str(n.id) == canonical_id]
    assert len(survivors) == 1
    node = survivors[0]  # the PERSISTED node, not the in-memory winner
    assert node.merged_aliases == ["Alicia"]
    assert node.merge_confidence == 0.93
    assert node.source_task == "canonicalize_entities"
    assert node.version == 2


@pytest.mark.asyncio
async def test_merge_keeps_winners_type_even_when_loser_traversed_first():
    person = EntityType(name="Person", description="a person")
    developer = EntityType(name="Developer", description="a developer")
    alice = Entity(name="Alice", description="engineer", is_a=person)
    alicia = Entity(name="Alicia", description="sw engineer", is_a=developer)
    # Loser (Alicia, is_a Developer) first; the survivor must still be a Person.
    summary = _make_summary(
        [
            (Edge(relationship_type="contains"), alicia),
            (Edge(relationship_type="contains"), alice),
        ]
    )
    judgments = [_judgment(0, True, "Alice", "merged", 0.95)]

    with _patch_vector_engine(), _patch_llm(judgments):
        await canonicalize_entities([summary])

    _nodes, edges = await _flatten([summary])
    canonical_id = str(Entity.id_for("Alice"))
    isa_targets = {str(e[1]) for e in edges if str(e[0]) == canonical_id and e[2] == "is_a"}
    assert isa_targets == {str(EntityType.id_for("Person"))}


@pytest.mark.asyncio
async def test_merge_id_is_bounded_to_a_member_not_llm_canonical_name():
    # A near-dup pair the judge merges, but the judge returns a canonical_name that
    # collides with an UNRELATED entity. The merged id must stay one of the pair's
    # own ids — never the unrelated entity's — so entities never compared are not
    # silently conflated.
    alice = _entity("Alice", "engineer")
    alicia = _entity("Alicia", "sw engineer")
    bob = _entity("Bob", "unrelated manager")  # orthogonal, never judged

    summary = _make_summary(
        [
            (Edge(relationship_type="contains"), alice),
            (Edge(relationship_type="contains"), alicia),
            (Edge(relationship_type="contains"), bob),
        ]
    )
    judgments = [_judgment(0, True, "Bob", "merged", 0.95)]  # adversarial canonical_name

    with _patch_vector_engine(), _patch_llm(judgments):
        await canonicalize_entities([summary])

    bob_id = str(Entity.id_for("Bob"))
    assert str(alice.id) == str(alicia.id)  # the near-dup pair collapsed
    assert str(alice.id) != bob_id  # but NOT onto the unrelated entity
    assert str(bob.id) == bob_id  # unrelated Bob untouched


@pytest.mark.asyncio
async def test_blank_reconciled_description_keeps_the_original():
    # A confirmed merge whose judge returned a blank reconciled description must
    # not wipe the survivor's description — keep the (richer) original instead.
    alice = _entity("Alice", "Senior engineer with 10 years of experience")
    alicia = _entity("Alicia", "sw engineer")
    summary = _make_summary(
        [
            (Edge(relationship_type="contains"), alice),
            (Edge(relationship_type="contains"), alicia),
        ]
    )
    judgments = [_judgment(0, True, "Alice", "   ", 0.95)]  # blank/whitespace description

    with _patch_vector_engine(), _patch_llm(judgments):
        await canonicalize_entities([summary])

    assert str(alice.id) == str(alicia.id)  # still merged
    assert alice.description == "Senior engineer with 10 years of experience"


# ---------------------------------------------------------------------------
# _select_winner determinism (order-independent)
# ---------------------------------------------------------------------------
def test_select_winner_exact_match_is_order_independent():
    alice = _entity("Alice")
    alicia = _entity("Alicia")
    members = [alice, alicia]

    forward = _select_winner(members, "Alice")
    backward = _select_winner(list(reversed(members)), "Alice")
    assert forward is alice
    assert backward is alice


def test_select_winner_falls_back_to_min_normalized_name():
    zeta = _entity("Zeta")
    alpha = _entity("Alpha")
    mid = _entity("Mid")
    members = [zeta, alpha, mid]

    # canonical matches none -> smallest normalized name ("alpha") wins.
    forward = _select_winner(members, "Brand New Name")
    backward = _select_winner(list(reversed(members)), "Brand New Name")
    assert forward is alpha
    assert backward is alpha
