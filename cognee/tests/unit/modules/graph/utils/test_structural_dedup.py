"""Tests for structural (graph-topology) dedup — issue #3630, Approach D.

Covers the shared fixture from the proposal: "Apple" / "Apple Inc." sharing
identical typed edges (structural duplicate), a founded_year contradiction,
and "Google" as a structurally distinct non-duplicate.

All tests are deterministic, zero LLM calls, no network.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline copies of the structural_resolution modules (avoids the full
# cognee import chain, mirrors the pattern used by test_provenance_mode.py)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import FrozenSet, Tuple, Iterable
from collections import defaultdict


@dataclass(frozen=True)
class StructuralFingerprint:
    node_id: str
    neighbor_ids: FrozenSet[str] = field(default_factory=frozenset)
    typed_edges: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)


def build_fingerprint(node_id, edges):
    neighbor_ids = set()
    typed_edges = set()
    for source_id, target_id, relationship_name in edges:
        if source_id == node_id and target_id != node_id:
            neighbor_ids.add(target_id)
            typed_edges.add((relationship_name, target_id))
        elif target_id == node_id and source_id != node_id:
            neighbor_ids.add(source_id)
            typed_edges.add((relationship_name, source_id))
    return StructuralFingerprint(node_id, frozenset(neighbor_ids), frozenset(typed_edges))


def structural_similarity(fp_a, fp_b):
    if not fp_a.typed_edges and not fp_b.typed_edges:
        return 0.0
    intersection = fp_a.typed_edges & fp_b.typed_edges
    union = fp_a.typed_edges | fp_b.typed_edges
    return len(intersection) / len(union) if union else 0.0


def generate_candidate_pairs(node_ids, node_types, edges, min_shared_neighbors=2):
    node_id_set = set(node_ids)
    neighbor_to_nodes = defaultdict(set)
    for source_id, target_id, _rel in edges:
        if source_id in node_id_set and target_id in node_id_set:
            neighbor_to_nodes[target_id].add(source_id)
            neighbor_to_nodes[source_id].add(target_id)

    pair_shared_count = defaultdict(int)
    for _neighbor, connected in neighbor_to_nodes.items():
        connected_list = sorted(connected)
        for i in range(len(connected_list)):
            for j in range(i + 1, len(connected_list)):
                a, b = connected_list[i], connected_list[j]
                pair_shared_count[(a, b)] += 1

    candidates = []
    for (a, b), count in pair_shared_count.items():
        if count < min_shared_neighbors:
            continue
        if node_types.get(a) != node_types.get(b):
            continue
        candidates.append((a, b))
    return candidates


@dataclass(frozen=True)
class Contradiction:
    field: str
    value_a: Any
    value_b: Any
    winner: Any
    winner_source: str
    reason: str


def detect_contradictions(edges_a, edges_b):
    contradictions = []
    for edge_a in edges_a:
        for edge_b in edges_b:
            same_rel = edge_a.get("relationship_name") == edge_b.get("relationship_name")
            same_target = edge_a.get("destination_node_id") == edge_b.get("destination_node_id")
            if not (same_rel and same_target):
                continue
            attrs_a = edge_a.get("attributes") or {}
            attrs_b = edge_b.get("attributes") or {}
            if attrs_a == attrs_b:
                continue
            created_a = edge_a.get("created_at", 0) or 0
            created_b = edge_b.get("created_at", 0) or 0
            if created_b >= created_a:
                winner_value, winner_source = attrs_b, "b"
            else:
                winner_value, winner_source = attrs_a, "a"
            contradictions.append(Contradiction(
                field=edge_a.get("relationship_name", ""),
                value_a=attrs_a, value_b=attrs_b,
                winner=winner_value, winner_source=winner_source, reason="recency",
            ))
    return contradictions


@dataclass
class MergeCandidate:
    node_a_id: str
    node_b_id: str
    similarity_score: float
    contradictions: List[Contradiction] = field(default_factory=list)


@dataclass
class MergeRecord:
    canonical_id: str
    merged_id: str
    similarity_score: float
    merge_reason: str
    contradictions: List[Dict[str, Any]]
    merged_at: int
    reversed_at: Optional[int] = None
    merged_node_snapshot: Optional[Dict[str, Any]] = None
    merged_node_edges_snapshot: Optional[List[Tuple[str, str, str, dict]]] = None


async def resolve_structural_duplicates(
    node_ids, node_types, edges, edge_lookup_by_node=None,
    similarity_threshold=0.7, min_shared_neighbors=2,
):
    simple_edges = [(s, t, r) for s, t, r, *_ in edges]
    candidate_pairs = generate_candidate_pairs(node_ids, node_types, simple_edges, min_shared_neighbors)
    results = []
    for a, b in candidate_pairs:
        fp_a = build_fingerprint(a, simple_edges)
        fp_b = build_fingerprint(b, simple_edges)
        score = structural_similarity(fp_a, fp_b)
        if score < similarity_threshold:
            continue
        contradictions = []
        if edge_lookup_by_node is not None:
            contradictions = detect_contradictions(
                edge_lookup_by_node.get(a, []), edge_lookup_by_node.get(b, [])
            )
        results.append(MergeCandidate(a, b, score, contradictions))
    return results


class FakeNode:
    def __init__(self, id, **kwargs):
        self.id = id
        self.source_node_ids = None
        self.merged_into = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self):
        return {"id": self.id, "source_node_ids": self.source_node_ids}


async def apply_structural_merges(nodes, edges, merge_candidates, now_ms):
    nodes_by_id = {str(n.id): n for n in nodes}
    merge_records = []
    canonical_of = {}

    for candidate in merge_candidates:
        canonical_id = candidate.node_a_id
        absorbed_id = candidate.node_b_id
        if absorbed_id in canonical_of or canonical_id in canonical_of:
            continue
        canonical_node = nodes_by_id.get(canonical_id)
        absorbed_node = nodes_by_id.get(absorbed_id)
        if canonical_node is None or absorbed_node is None:
            continue

        existing = getattr(canonical_node, "source_node_ids", None) or []
        if absorbed_id not in existing:
            existing = [*existing, absorbed_id]
        canonical_node.source_node_ids = existing
        absorbed_node.merged_into = canonical_id
        canonical_of[absorbed_id] = canonical_id

        merge_records.append(MergeRecord(
            canonical_id=canonical_id, merged_id=absorbed_id,
            similarity_score=candidate.similarity_score,
            merge_reason="structural_overlap",
            contradictions=[
                {"field": c.field, "kept": c.winner,
                 "discarded": c.value_a if c.winner_source == "b" else c.value_b,
                 "why": c.reason}
                for c in candidate.contradictions
            ],
            merged_at=now_ms,
            merged_node_snapshot=absorbed_node.model_dump(),
            merged_node_edges_snapshot=[e for e in edges if e[0] == absorbed_id or e[1] == absorbed_id],
        ))

    updated_edges = []
    for source_id, target_id, rel, attrs in edges:
        updated_edges.append((canonical_of.get(source_id, source_id), canonical_of.get(target_id, target_id), rel, attrs))

    return list(nodes_by_id.values()), updated_edges, merge_records


def undo_merge(merge_record, nodes, edges, now_ms):
    if merge_record.reversed_at is not None:
        raise ValueError("Already reversed.")
    if merge_record.merged_node_snapshot is None:
        raise ValueError("No snapshot to restore from.")

    canonical_id = merge_record.canonical_id
    absorbed_id = merge_record.merged_id
    nodes_by_id = {str(n.id): n for n in nodes}
    absorbed_node = nodes_by_id.get(absorbed_id)
    canonical_node = nodes_by_id.get(canonical_id)

    if absorbed_node is not None:
        absorbed_node.merged_into = None
    if canonical_node is not None:
        existing = getattr(canonical_node, "source_node_ids", None) or []
        canonical_node.source_node_ids = [s for s in existing if s != absorbed_id]

    restored_edges = list(edges)
    if merge_record.merged_node_edges_snapshot:
        snapshot_edges = merge_record.merged_node_edges_snapshot

        def is_repointed(edge):
            source_id, target_id, rel, _attrs = edge
            for snap_s, snap_t, snap_r, _ in snapshot_edges:
                if snap_r != rel:
                    continue
                src_match = (source_id == canonical_id and snap_s == absorbed_id) or source_id == snap_s
                tgt_match = (target_id == canonical_id and snap_t == absorbed_id) or target_id == snap_t
                if src_match and tgt_match:
                    return True
            return False

        restored_edges = [e for e in restored_edges if not is_repointed(e)]
        restored_edges.extend(snapshot_edges)

    merge_record.reversed_at = now_ms
    return list(nodes_by_id.values()), restored_edges, merge_record


# ===========================================================================
# Shared fixture (from the proposal)
# ===========================================================================

def make_fixture():
    nodes = [
        FakeNode("Apple", type="Entity"),
        FakeNode("Apple Inc", type="Entity"),
        FakeNode("Tim Cook", type="Entity"),
        FakeNode("iPhone", type="Entity"),
        FakeNode("Google", type="Entity"),
    ]
    node_types = {n.id: "Entity" for n in nodes}
    edges = [
        ("Apple", "Tim Cook", "CEO", {}),
        ("Apple Inc", "Tim Cook", "CEO", {}),
        ("Apple", "iPhone", "MAKES", {}),
        ("Apple Inc", "iPhone", "MAKES", {}),
    ]
    edge_lookup = {
        "Apple": [
            {"relationship_name": "founded_year", "destination_node_id": "iPhone",
             "attributes": {"value": "1976"}, "created_at": 1000},
        ],
        "Apple Inc": [
            {"relationship_name": "founded_year", "destination_node_id": "iPhone",
             "attributes": {"value": "1977"}, "created_at": 2000},
        ],
    }
    return nodes, node_types, edges, edge_lookup


# ===========================================================================
# Tests
# ===========================================================================

class TestStructuralDedupMergesIdenticalTopology:
    def test_apple_variants_merge(self):
        nodes, node_types, edges, _ = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        assert len(candidates) == 1
        assert {candidates[0].node_a_id, candidates[0].node_b_id} == {"Apple", "Apple Inc"}

    def test_similarity_score_is_high(self):
        nodes, node_types, edges, _ = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        assert candidates[0].similarity_score >= 0.7


class TestStructuralDedupSkipsLowSimilarity:
    def test_google_not_merged(self):
        """Google shares no edges with Apple/Apple Inc — must not be flagged."""
        nodes, node_types, edges, _ = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        involved = {c.node_a_id for c in candidates} | {c.node_b_id for c in candidates}
        assert "Google" not in involved


class TestContradictionFlaggedAndResolvedByRecency:
    def test_contradiction_detected(self):
        nodes, node_types, edges, edge_lookup = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
            edge_lookup_by_node=edge_lookup,
        ))
        assert len(candidates[0].contradictions) == 1

    def test_recency_winner_is_1977(self):
        nodes, node_types, edges, edge_lookup = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
            edge_lookup_by_node=edge_lookup,
        ))
        contradiction = candidates[0].contradictions[0]
        assert contradiction.winner == {"value": "1977"}
        assert contradiction.reason == "recency"

    def test_loser_value_preserved_not_discarded(self):
        nodes, node_types, edges, edge_lookup = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
            edge_lookup_by_node=edge_lookup,
        ))
        contradiction = candidates[0].contradictions[0]
        # the loser (1976) must still be present in the record, not dropped
        assert {"value": "1976"} in (contradiction.value_a, contradiction.value_b)


class TestMergeProvenancePreserved:
    def test_source_node_ids_contains_absorbed(self):
        nodes, node_types, edges, edge_lookup = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
            edge_lookup_by_node=edge_lookup,
        ))
        updated_nodes, _, records = asyncio.run(
            apply_structural_merges(nodes, edges, candidates, now_ms=1000)
        )
        canonical = next(n for n in updated_nodes if n.id == "Apple")
        assert "Apple Inc" in canonical.source_node_ids

    def test_merge_record_has_reason_and_score(self):
        nodes, node_types, edges, _ = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        _, _, records = asyncio.run(apply_structural_merges(nodes, edges, candidates, now_ms=1000))
        assert records[0].merge_reason == "structural_overlap"
        assert records[0].similarity_score >= 0.7

    def test_absorbed_node_marked_not_deleted(self):
        """Merges are non-destructive — absorbed node stays, tagged merged_into."""
        nodes, node_types, edges, _ = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        updated_nodes, _, _ = asyncio.run(apply_structural_merges(nodes, edges, candidates, now_ms=1000))
        absorbed = next(n for n in updated_nodes if n.id == "Apple Inc")
        assert absorbed.merged_into == "Apple"
        assert absorbed in updated_nodes  # still present, not deleted


class TestMergeIsReversible:
    def test_undo_restores_source_node_ids(self):
        nodes, node_types, edges, _ = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        updated_nodes, updated_edges, records = asyncio.run(
            apply_structural_merges(nodes, edges, candidates, now_ms=1000)
        )
        restored_nodes, restored_edges, record = undo_merge(records[0], updated_nodes, updated_edges, now_ms=2000)
        canonical = next(n for n in restored_nodes if n.id == "Apple")
        assert "Apple Inc" not in canonical.source_node_ids

    def test_undo_clears_merged_into(self):
        nodes, node_types, edges, _ = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        updated_nodes, updated_edges, records = asyncio.run(
            apply_structural_merges(nodes, edges, candidates, now_ms=1000)
        )
        restored_nodes, _, _ = undo_merge(records[0], updated_nodes, updated_edges, now_ms=2000)
        absorbed = next(n for n in restored_nodes if n.id == "Apple Inc")
        assert absorbed.merged_into is None

    def test_undo_stamps_reversed_at(self):
        nodes, node_types, edges, _ = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        _, _, records = asyncio.run(apply_structural_merges(nodes, edges, candidates, now_ms=1000))
        _, _, record = undo_merge(records[0], nodes, edges, now_ms=2000)
        assert record.reversed_at == 2000

    def test_double_undo_raises(self):
        nodes, node_types, edges, _ = make_fixture()
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        updated_nodes, updated_edges, records = asyncio.run(
            apply_structural_merges(nodes, edges, candidates, now_ms=1000)
        )
        undo_merge(records[0], updated_nodes, updated_edges, now_ms=2000)
        with pytest.raises(ValueError):
            undo_merge(records[0], updated_nodes, updated_edges, now_ms=3000)


class TestIdentityOnlyNodesUntouched:
    def test_low_similarity_nodes_not_merged(self):
        """Nodes below threshold must remain completely untouched."""
        nodes = [FakeNode("A", type="Entity"), FakeNode("B", type="Entity"), FakeNode("X", type="Entity")]
        node_types = {"A": "Entity", "B": "Entity", "X": "Entity"}
        # A and B share only 1 neighbor (below min_shared_neighbors default of 2)
        edges = [("A", "X", "REL1", {})]
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        assert candidates == []

    def test_type_gating_prevents_cross_type_merge(self):
        """Entity and DocumentChunk sharing neighbors must never merge."""
        nodes = [
            FakeNode("Entity1", type="Entity"),
            FakeNode("Chunk1", type="DocumentChunk"),
            FakeNode("Neighbor1", type="Entity"),
            FakeNode("Neighbor2", type="Entity"),
        ]
        node_types = {"Entity1": "Entity", "Chunk1": "DocumentChunk",
                      "Neighbor1": "Entity", "Neighbor2": "Entity"}
        edges = [
            ("Entity1", "Neighbor1", "REL", {}),
            ("Chunk1", "Neighbor1", "REL", {}),
            ("Entity1", "Neighbor2", "REL", {}),
            ("Chunk1", "Neighbor2", "REL", {}),
        ]
        candidates = asyncio.run(resolve_structural_duplicates(
            node_ids=[n.id for n in nodes], node_types=node_types, edges=edges,
        ))
        involved_pairs = [{c.node_a_id, c.node_b_id} for c in candidates]
        assert {"Entity1", "Chunk1"} not in involved_pairs


class TestFingerprintAndSimilarity:
    def test_fingerprint_empty_for_isolated_node(self):
        fp = build_fingerprint("Isolated", [("A", "B", "REL", {})[:3]])
        assert fp.typed_edges == frozenset()

    def test_similarity_zero_for_disjoint_fingerprints(self):
        fp_a = build_fingerprint("A", [("A", "X", "REL1")])
        fp_b = build_fingerprint("B", [("B", "Y", "REL2")])
        assert structural_similarity(fp_a, fp_b) == 0.0

    def test_similarity_one_for_identical_fingerprints(self):
        edges = [("A", "X", "REL1"), ("B", "X", "REL1")]
        fp_a = build_fingerprint("A", edges)
        fp_b = build_fingerprint("B", edges)
        assert structural_similarity(fp_a, fp_b) == 1.0

    def test_typed_edges_prevent_false_positive(self):
        """Same neighbor via different relationship types must NOT score as similar."""
        edges = [("A", "X", "MAKES"), ("B", "X", "COMPETES_WITH")]
        fp_a = build_fingerprint("A", edges)
        fp_b = build_fingerprint("B", edges)
        assert structural_similarity(fp_a, fp_b) == 0.0