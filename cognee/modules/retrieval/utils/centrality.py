"""Pure-Python centrality over a projected ``CogneeGraph`` memory fragment.

Issue #3378: rank results by structural graph importance (PageRank / degree)
rather than vector similarity alone. Per the maintainer constraint on that
issue, centrality is computed directly on the native projected fragment
(``CogneeGraph`` from ``project_graph_from_db`` / ``project_neighborhood_from_db``)
using its already-materialized adjacency — no networkx or Neo4j GDS dependency.

A query-biased ``personalization`` vector lets a single code path serve both
cases the acceptance criteria ask for: with query anchors the teleport mass
concentrates on the seed nodes (importance relative to the question); with no
anchor it is uniform, which is plain global PageRank.
"""

from typing import Dict, Iterable, List, Optional

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph

# EntityType nodes are internal taxonomy labels that every Entity points to via
# an ``is_a`` edge, so they dominate any degree/PageRank ranking while carrying
# no query value. Mirror get_schema_inventory's hygiene and drop them by default.
INTERNAL_NODE_TYPES = frozenset({"EntityType"})


def select_rankable_node_ids(
    graph: CogneeGraph,
    exclude_types: Iterable[str] = INTERNAL_NODE_TYPES,
) -> List[str]:
    """Return node ids eligible for ranking, dropping internal taxonomy labels."""
    excluded = set(exclude_types)
    return [
        node_id
        for node_id, node in graph.nodes.items()
        if node.attributes.get("type") not in excluded
    ]


def _build_directed_adjacency(graph: CogneeGraph, node_ids: List[str]):
    """Out- and in-neighbor sets restricted to ``node_ids`` (self-loops dropped).

    Undirected edges contribute in both directions; directed edges follow
    ``node1 -> node2``. Sets dedupe parallel edges so a multi-edge pair counts
    once, which is what degree/PageRank over a simple graph expect.
    """
    ids = set(node_ids)
    out_neighbors: Dict[str, set] = {nid: set() for nid in ids}
    in_neighbors: Dict[str, set] = {nid: set() for nid in ids}
    for edge in graph.edges:
        source = edge.node1.id
        target = edge.node2.id
        if source not in ids or target not in ids or source == target:
            continue
        out_neighbors[source].add(target)
        in_neighbors[target].add(source)
        if not edge.directed:
            out_neighbors[target].add(source)
            in_neighbors[source].add(target)
    return out_neighbors, in_neighbors


def degree_centrality(
    graph: CogneeGraph,
    node_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Distinct-neighbor degree centrality, normalized by ``N - 1``.

    Direction-agnostic (counts any incident neighbor) — a cheap, always-safe
    structural signal computed straight from the projection.
    """
    if node_ids is None:
        node_ids = list(graph.nodes.keys())
    ids = list(dict.fromkeys(node_ids))  # stable de-dup
    n = len(ids)
    if n < 2:
        return {nid: 0.0 for nid in ids}

    out_neighbors, in_neighbors = _build_directed_adjacency(graph, ids)
    norm = float(n - 1)
    return {nid: len(out_neighbors[nid] | in_neighbors[nid]) / norm for nid in ids}


def pagerank(
    graph: CogneeGraph,
    node_ids: Optional[List[str]] = None,
    personalization: Optional[Dict[str, float]] = None,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> Dict[str, float]:
    """Directed PageRank via power iteration on the projected adjacency.

    - Runs on a directed view (``CogneeGraph.directed`` is True by default);
      undirected edges contribute both directions.
    - Dangling nodes (no out-links) redistribute their mass over the teleport
      distribution each iteration, so total rank is conserved.
    - ``personalization`` biases the teleport vector toward query-seed nodes;
      when ``None`` the teleport is uniform, i.e. plain global PageRank.

    Scores sum to 1.0 and are deterministic for a given projection.
    """
    if node_ids is None:
        node_ids = list(graph.nodes.keys())
    ids = list(dict.fromkeys(node_ids))
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: 1.0}

    out_neighbors, _ = _build_directed_adjacency(graph, ids)

    # Teleport / personalization distribution (normalized, restricted to ids).
    if personalization:
        weights = {nid: max(0.0, float(personalization.get(nid, 0.0))) for nid in ids}
        total = sum(weights.values())
        teleport = (
            {nid: weights[nid] / total for nid in ids}
            if total > 0
            else {nid: 1.0 / n for nid in ids}
        )
    else:
        teleport = {nid: 1.0 / n for nid in ids}

    rank = {nid: 1.0 / n for nid in ids}
    dangling_nodes = [nid for nid in ids if not out_neighbors[nid]]

    for _ in range(max_iter):
        dangling_mass = damping * sum(rank[nid] for nid in dangling_nodes)
        # Base mass: random-surfer teleport plus the redistributed dangling mass.
        new_rank = {
            nid: (1.0 - damping) * teleport[nid] + dangling_mass * teleport[nid] for nid in ids
        }
        for nid in ids:
            targets = out_neighbors[nid]
            if not targets:
                continue
            share = damping * rank[nid] / len(targets)
            for target in targets:
                new_rank[target] += share
        delta = sum(abs(new_rank[nid] - rank[nid]) for nid in ids)
        rank = new_rank
        if delta < tol:
            break
    return rank


def rank_top_k(scores: Dict[str, float], top_k: Optional[int]) -> List[str]:
    """Top-k node ids by score, descending, breaking ties on id for determinism."""
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None and top_k > 0:
        ordered = ordered[:top_k]
    return [node_id for node_id, _ in ordered]
