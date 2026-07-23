"""Structural (graph-topology) duplicate resolution — issue #3630, Approach D.

Two entities extracted under different surface names — "Apple" and "Apple Inc." —
are often the same real-world thing when they share the same neighbours through
the same relationships: both point to "Tim Cook" via ``CEO`` and to "iPhone" via
``MAKES``. This module finds such structural duplicates within an ingestion batch
and links each pair with a ``merged_into`` edge (duplicate -> canonical), so the
duplication is recorded in the graph itself — provenance-preserving,
non-destructive (both nodes and all their edges are kept), and reversible (drop
the ``merged_into`` edge to undo).

Detection is deterministic — typed Jaccard over each node's
``{(relationship_name, neighbour_id)}`` signature, gated to same-type nodes that
already share neighbours — so it stays reproducible under the mocked-LLM CI
harness (#3601), with no LLM call. Typed (rather than plain neighbour) overlap
is what stops a false positive when two nodes touch the same neighbour through
*different* relationships (``Apple --MAKES--> iPhone`` vs
``Google --COMPETES_WITH--> iPhone``).

It extends the existing dedup path — called from ``add_data_points`` right after
``deduplicate_nodes_and_edges`` — rather than forking a new pipeline, and is
opt-in via ``add_data_points(..., structural_dedup=True)``. Consuming
``merged_into`` at query time (collapsing the duplicates during retrieval) is
deliberately left to a follow-up, once the sibling approaches (A–E) are compared
on a shared evaluation fixture.
"""

from collections import defaultdict
from typing import Any

# Graph edges flow through the pipeline as
# (source_node_id, target_node_id, relationship_name, properties) tuples.
Edge = tuple[Any, Any, str, dict]

# The synthetic relationship that records a structural merge.
MERGED_INTO_RELATIONSHIP = "merged_into"

# A pair merges only when its typed neighbourhoods overlap at least this much.
STRUCTURAL_DEDUP_THRESHOLD = 0.7
# ... and only pairs already sharing at least this many neighbours are scored,
# which keeps the pass ~O(E) instead of O(n^2) pairwise comparisons.
STRUCTURAL_DEDUP_MIN_SHARED_NEIGHBORS = 2


def resolve_structural_duplicates(
    nodes: list[Any],
    edges: list[Edge],
    *,
    similarity_threshold: float = STRUCTURAL_DEDUP_THRESHOLD,
    min_shared_neighbors: int = STRUCTURAL_DEDUP_MIN_SHARED_NEIGHBORS,
) -> list[Edge]:
    """Link structurally-duplicate nodes in a batch with ``merged_into`` edges.

    Args:
        nodes: The batch's nodes (``DataPoint``s), after identity dedup.
        edges: ``(source, target, relationship_name, properties)`` tuples.
        similarity_threshold: Minimum typed-Jaccard score for two same-type
            nodes to count as duplicates.
        min_shared_neighbors: Only score pairs that already share at least this
            many neighbours — cheap blocking that avoids full pairwise scoring.

    Returns:
        New ``merged_into`` edges (duplicate -> canonical), one per detected
        duplicate pair, each carrying ``similarity_score`` and ``resolution`` in
        its properties. Empty when nothing crosses the threshold. ``nodes`` and
        ``edges`` (and their dicts) are never mutated.
    """
    node_by_id = {str(node.id): node for node in nodes}
    if len(node_by_id) < 2:
        return []

    node_types = {node_id: _node_type(node) for node_id, node in node_by_id.items()}

    # Single pass over edges: build each node's typed neighbourhood and the set
    # of neighbours it touches. Only edges between two batch nodes count; a
    # self-loop is not adjacency evidence.
    typed_neighbourhood: dict[str, set[tuple[str, str]]] = {
        node_id: set() for node_id in node_by_id
    }
    neighbours_of: dict[str, set[str]] = {node_id: set() for node_id in node_by_id}
    for source, target, relationship_name, *_ in edges:
        source_id, target_id = str(source), str(target)
        if source_id == target_id or source_id not in node_by_id or target_id not in node_by_id:
            continue
        typed_neighbourhood[source_id].add((relationship_name, target_id))
        typed_neighbourhood[target_id].add((relationship_name, source_id))
        neighbours_of[source_id].add(target_id)
        neighbours_of[target_id].add(source_id)

    # Score each blocked candidate; keep those above the threshold.
    scored: list[tuple[float, str, str]] = []
    for node_a, node_b in _candidate_pairs(neighbours_of, node_types, min_shared_neighbors):
        score = _typed_jaccard(typed_neighbourhood[node_a], typed_neighbourhood[node_b])
        if score >= similarity_threshold:
            scored.append((score, node_a, node_b))

    # Highest similarity first (ties broken by id) so each node links to its best
    # match; greedily skip any node already resolved, which keeps merges a clean
    # forest (a canonical may absorb several duplicates, but no chains form).
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    resolved: set[str] = set()
    merge_edges: list[Edge] = []
    for score, node_a, node_b in scored:
        if node_a in resolved or node_b in resolved:
            continue
        duplicate_id, canonical_id = _orient(node_a, node_b, neighbours_of)
        resolved.add(duplicate_id)
        merge_edges.append(
            (
                node_by_id[duplicate_id].id,
                node_by_id[canonical_id].id,
                MERGED_INTO_RELATIONSHIP,
                {"similarity_score": round(score, 4), "resolution": "structural_overlap"},
            )
        )
    return merge_edges


def _node_type(node: Any) -> str:
    """The node's type label used for type-gated blocking (Entity vs DocumentChunk)."""
    return getattr(node, "type", None) or type(node).__name__


def _candidate_pairs(
    neighbours_of: dict[str, set[str]],
    node_types: dict[str, str],
    min_shared_neighbors: int,
):
    """Yield same-type node pairs that share >= ``min_shared_neighbors`` neighbours.

    Built from a neighbour -> nodes inverted index, so cost scales with the
    number of co-occurrences rather than the square of the node count.
    """
    inverted: dict[str, list[str]] = defaultdict(list)
    for node_id, neighbours in neighbours_of.items():
        for neighbour in neighbours:
            inverted[neighbour].append(node_id)

    shared_count: dict[tuple[str, str], int] = defaultdict(int)
    for members in inverted.values():
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                shared_count[(members[i], members[j])] += 1

    for (node_a, node_b), count in shared_count.items():
        if count >= min_shared_neighbors and node_types.get(node_a) == node_types.get(node_b):
            yield node_a, node_b


def _typed_jaccard(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
    """Jaccard overlap of two typed neighbourhoods, in ``[0.0, 1.0]``."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _orient(node_a: str, node_b: str, neighbours_of: dict[str, set[str]]) -> tuple[str, str]:
    """Return ``(duplicate_id, canonical_id)``.

    The better-connected node is kept as canonical (it carries more evidence);
    ties break on the smaller id, so the choice is fully deterministic.
    """
    degree_a, degree_b = len(neighbours_of[node_a]), len(neighbours_of[node_b])
    if degree_a != degree_b:
        canonical = node_a if degree_a > degree_b else node_b
    else:
        canonical = min(node_a, node_b)
    duplicate = node_b if canonical == node_a else node_a
    return duplicate, canonical
