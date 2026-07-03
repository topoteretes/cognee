"""Merge scoring + execution for structural dedup (issue #3630, Approach D).

Ties together fingerprinting, candidate generation, and contradiction
detection into the end-to-end resolve -> apply flow described in the
issue proposal:

    deduplicated = deduplicate_nodes_and_edges(nodes, edges)           # existing (identity)
    merge_candidates = await resolve_structural_duplicates(...)        # new
    final = await apply_structural_merges(..., merge_candidates)       # new

Merges are non-destructive: the canonical (surviving) node accumulates
`source_node_ids` from every node folded into it, and every merge is
recorded so it can be audited or undone.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

from .fingerprint import build_fingerprint, structural_similarity
from .candidate_generation import generate_candidate_pairs
from .contradiction_detection import detect_contradictions, Contradiction


DEFAULT_SIMILARITY_THRESHOLD = 0.7


@dataclass
class MergeCandidate:
    """A scored pair of nodes proposed for merging."""

    node_a_id: str
    node_b_id: str
    similarity_score: float
    contradictions: List[Contradiction] = field(default_factory=list)


@dataclass
class MergeRecord:
    """Audit trail entry for a completed (or undone) merge.

    Mirrors the SQLAlchemy MergeRecord model shape described in the
    proposal; kept as a plain dataclass here so structural resolution has
    no hard dependency on the persistence layer and can be unit-tested
    in isolation.
    """

    canonical_id: str
    merged_id: str
    similarity_score: float
    merge_reason: str
    contradictions: List[Dict[str, Any]]
    merged_at: int
    reversed_at: Optional[int] = None

    # Snapshot of the absorbed node, needed to reverse the merge.
    merged_node_snapshot: Optional[Dict[str, Any]] = None
    merged_node_edges_snapshot: Optional[List[Tuple[str, str, str, dict]]] = None


async def resolve_structural_duplicates(
    node_ids: List[str],
    node_types: Dict[str, str],
    edges: List[Tuple[str, str, str, dict]],
    edge_lookup_by_node: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_shared_neighbors: int = 2,
) -> List[MergeCandidate]:
    """Score candidate node pairs for structural duplication / contradiction.

    Parameters
    ----------
    node_ids : List[str]
        Node ids surviving identity-based dedup.
    node_types : Dict[str, str]
        node_id -> type, used for type-gated blocking.
    edges : List[Tuple[str, str, str, dict]]
        (source_id, target_id, relationship_name, attributes) tuples for
        the in-memory batch (pre-persistence). Attributes may be empty.
    edge_lookup_by_node : Optional[Dict[str, List[Dict]]]
        Optional pre-built map of node_id -> list of edge dicts (with
        relationship_name, destination_node_id, attributes, created_at)
        used for contradiction detection. If omitted, contradiction
        detection is skipped (structural score is still computed).
    similarity_threshold : float
        Minimum typed-Jaccard score to propose a merge.
    min_shared_neighbors : int
        Passed through to candidate generation.

    Returns
    -------
    List[MergeCandidate]
    """
    simple_edges = [(s, t, r) for s, t, r, *_ in edges]

    candidate_pairs = generate_candidate_pairs(
        node_ids=node_ids,
        node_types=node_types,
        edges=simple_edges,
        min_shared_neighbors=min_shared_neighbors,
    )

    results: List[MergeCandidate] = []
    for node_a_id, node_b_id in candidate_pairs:
        fp_a = build_fingerprint(node_a_id, simple_edges)
        fp_b = build_fingerprint(node_b_id, simple_edges)
        score = structural_similarity(fp_a, fp_b)

        if score < similarity_threshold:
            continue

        contradictions: List[Contradiction] = []
        if edge_lookup_by_node is not None:
            edges_a = edge_lookup_by_node.get(node_a_id, [])
            edges_b = edge_lookup_by_node.get(node_b_id, [])
            contradictions = detect_contradictions(edges_a, edges_b)

        results.append(
            MergeCandidate(
                node_a_id=node_a_id,
                node_b_id=node_b_id,
                similarity_score=score,
                contradictions=contradictions,
            )
        )

    return results


async def apply_structural_merges(
    nodes: List[Any],
    edges: List[Tuple[str, str, str, dict]],
    merge_candidates: List[MergeCandidate],
    now_ms: int,
) -> Tuple[List[Any], List[Tuple[str, str, str, dict]], List[MergeRecord]]:
    """Apply merges to in-memory nodes/edges lists, non-destructively.

    The canonical node (node_a in each candidate pair, by convention "first
    seen wins") absorbs the merged node: it gains a `source_node_ids` entry
    for the absorbed node's id, and every edge referencing the absorbed
    node is repointed to the canonical node's id.

    The absorbed node is *not* deleted from `nodes` — cognee's storage
    layer preserves it with a `merged_into` marker so the merge is
    reversible and auditable (see `undo.py`).

    Returns
    -------
    Tuple of (updated_nodes, updated_edges, merge_records)
    """
    nodes_by_id = {str(n.id): n for n in nodes}
    merge_records: List[MergeRecord] = []

    # canonical_of[absorbed_id] = canonical_id
    canonical_of: Dict[str, str] = {}

    for candidate in merge_candidates:
        canonical_id = candidate.node_a_id
        absorbed_id = candidate.node_b_id

        # Skip if either side was already merged into something else this pass.
        if absorbed_id in canonical_of or canonical_id in canonical_of:
            continue

        canonical_node = nodes_by_id.get(canonical_id)
        absorbed_node = nodes_by_id.get(absorbed_id)
        if canonical_node is None or absorbed_node is None:
            continue

        # Track absorbed source ids on the canonical node (non-destructive).
        existing_sources = getattr(canonical_node, "source_node_ids", None) or []
        if absorbed_id not in existing_sources:
            existing_sources = [*existing_sources, absorbed_id]
        try:
            canonical_node.source_node_ids = existing_sources
        except (AttributeError, ValueError):
            # Model may not declare this field; attach dynamically for audit.
            object.__setattr__(canonical_node, "source_node_ids", existing_sources)

        # Mark absorbed node as merged (non-destructive: node stays in the list).
        try:
            absorbed_node.merged_into = canonical_id
        except (AttributeError, ValueError):
            object.__setattr__(absorbed_node, "merged_into", canonical_id)

        canonical_of[absorbed_id] = canonical_id

        merge_records.append(
            MergeRecord(
                canonical_id=canonical_id,
                merged_id=absorbed_id,
                similarity_score=candidate.similarity_score,
                merge_reason="structural_overlap",
                contradictions=[
                    {
                        "field": c.field,
                        "kept": c.winner,
                        "discarded": c.value_a if c.winner_source == "b" else c.value_b,
                        "why": c.reason,
                    }
                    for c in candidate.contradictions
                ],
                merged_at=now_ms,
                merged_node_snapshot=absorbed_node.model_dump()
                if hasattr(absorbed_node, "model_dump")
                else dict(vars(absorbed_node)),
                merged_node_edges_snapshot=[
                    e for e in edges if e[0] == absorbed_id or e[1] == absorbed_id
                ],
            )
        )

    # Repoint edges from absorbed ids to their canonical ids.
    updated_edges = []
    for source_id, target_id, relationship_name, attributes in edges:
        new_source = canonical_of.get(source_id, source_id)
        new_target = canonical_of.get(target_id, target_id)
        updated_edges.append((new_source, new_target, relationship_name, attributes))

    return list(nodes_by_id.values()), updated_edges, merge_records