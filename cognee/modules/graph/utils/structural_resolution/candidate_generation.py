"""Candidate pair generation for structural dedup (issue #3630, Approach D).

Full O(n^2) pairwise comparison across all nodes is prohibitive. This module
narrows the search space to a small set of *candidate pairs* worth scoring:

1. Type-gated blocking — only compare nodes with the same `type`. An Entity
   is never a structural duplicate of a DocumentChunk.
2. Shared-neighbor inverted index — build neighbor -> {node_ids} once, then
   any two nodes that share >= `min_shared_neighbors` neighbors become a
   candidate pair. This is O(E) to build and produces far fewer pairs than
   O(n^2) for real graphs.
3. Skip nodes that already share an identity (already merged by the
   upstream identity-based dedup pass) — structural resolution only runs on
   nodes that survived identity dedup.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple


def generate_candidate_pairs(
    node_ids: Iterable[str],
    node_types: Dict[str, str],
    edges: Iterable[Tuple[str, str, str]],
    min_shared_neighbors: int = 2,
) -> List[Tuple[str, str]]:
    """Generate candidate node-id pairs worth scoring for structural similarity.

    Parameters
    ----------
    node_ids : Iterable[str]
        All node ids under consideration (post identity-dedup).
    node_types : Dict[str, str]
        Mapping of node_id -> type label, used for type-gated blocking.
    edges : Iterable[Tuple[str, str, str]]
        (source_id, target_id, relationship_name) triples.
    min_shared_neighbors : int
        Minimum number of shared neighbors for a pair to be considered a
        candidate. Higher = fewer, higher-precision candidates.

    Returns
    -------
    List[Tuple[str, str]]
        Deduplicated, order-independent candidate pairs (node_a, node_b)
        with node_a < node_b lexicographically, so each pair appears once.
    """
    node_id_set = set(node_ids)

    # Build inverted index: neighbor_id -> set of node_ids that connect to it
    neighbor_to_nodes: Dict[str, Set[str]] = defaultdict(set)
    for source_id, target_id, _relationship_name in edges:
        if source_id in node_id_set and target_id in node_id_set:
            # both endpoints are nodes we're deduping over
            neighbor_to_nodes[target_id].add(source_id)
            neighbor_to_nodes[source_id].add(target_id)

    # Count shared-neighbor co-occurrences via the inverted index
    pair_shared_count: Dict[Tuple[str, str], int] = defaultdict(int)
    for _neighbor_id, connected_nodes in neighbor_to_nodes.items():
        connected_list = sorted(connected_nodes)
        for i in range(len(connected_list)):
            for j in range(i + 1, len(connected_list)):
                a, b = connected_list[i], connected_list[j]
                pair_shared_count[(a, b)] += 1

    candidates: List[Tuple[str, str]] = []
    for (a, b), shared_count in pair_shared_count.items():
        if shared_count < min_shared_neighbors:
            continue
        # Type-gated blocking: only compare same-type nodes
        if node_types.get(a) != node_types.get(b):
            continue
        candidates.append((a, b))

    return candidates