"""Bounded subgraph selection for graph visualization.

``visualize_graph()`` renders a bounded subgraph by default instead of the whole
graph. Seeds are resolved by priority — explicit node ids, a recall/search
result's graph provenance, or a query string — and, when none of those are
given, the highest-degree nodes so a bare call still shows a representative
view. The seed neighborhood is expanded through the shared
``graph_engine.get_neighborhood()`` primitive and capped at ``max_nodes``.
Pass ``full=True`` to render the entire graph (legacy behavior).
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from cognee.infrastructure.databases.graph.graph_db_interface import EdgeData, Node
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch
from cognee.shared.logging_utils import get_logger

logger = get_logger("visualization.subgraph_data")

DEFAULT_NEIGHBORHOOD_DEPTH = 2
DEFAULT_SEED_TOP_K = 10
DEFAULT_MAX_NODES = 500
DEFAULT_WIDE_SEARCH_TOP_K = 100

# Node vector collections searched to turn a query into seed nodes — the same
# node-bearing collections search uses (keep in sync with the default list in
# retrieval/utils/brute_force_triplet_search.py). Edge collections are
# intentionally excluded: their hits are not graph nodes and cannot seed a
# neighborhood.
_SEED_VECTOR_COLLECTIONS = [
    "Entity_name",
    "TextSummary_text",
    "EntityType_name",
    "DocumentChunk_text",
]

# (nodes, edges) in the shape get_graph_data()/get_neighborhood() already return.
GraphData = Tuple[List[Node], List[EdgeData]]


def _unique_preserve_order(node_ids: List[str]) -> List[str]:
    """Order-preserving de-duplication of stringified node ids."""
    return list(dict.fromkeys(str(node_id) for node_id in node_ids))


def _coerce_node_ids(value: Any) -> List[str]:
    """Node ids from a value that is either a ``node_ids`` list or a provenance
    mapping (``{"node_ids": [...]}``). Anything else yields ``[]``."""
    if isinstance(value, dict):
        value = value.get("node_ids")
    if isinstance(value, (list, tuple)):
        return [str(n) for n in value]
    return []


def resolve_seeds_from_recall(recall_result: Any) -> List[str]:
    """Seed node ids from a recall/search result's graph provenance.

    Handles the shapes cognee results actually carry node ids in: a mapping with
    a ``node_ids`` list (a citation/provenance payload), and recall()/remember()
    QA entries — as objects or dicts — that expose ``used_graph_element_ids``.
    Any other shape yields ``[]`` rather than raising.
    """
    if recall_result is None:
        return []

    if isinstance(recall_result, dict):
        seeds = _coerce_node_ids(recall_result.get("node_ids")) or _coerce_node_ids(
            recall_result.get("used_graph_element_ids")
        )
        return _unique_preserve_order(seeds)

    items = recall_result if isinstance(recall_result, (list, tuple)) else [recall_result]
    node_ids: List[str] = []
    for item in items:
        used = (
            item.get("used_graph_element_ids")
            if isinstance(item, dict)
            else getattr(item, "used_graph_element_ids", None)
        )
        node_ids.extend(_coerce_node_ids(used))
    return _unique_preserve_order(node_ids)


async def resolve_seeds_from_query(
    query: str,
    seed_top_k: int = DEFAULT_SEED_TOP_K,
    wide_search_top_k: int = DEFAULT_WIDE_SEARCH_TOP_K,
) -> List[str]:
    """Seed node ids for a query, ranked by vector distance (nearest first).

    ``NodeEdgeVectorSearch.extract_relevant_node_ids()`` returns an unordered
    set, so a plain slice of it is neither top-k nor deterministic. We rank the
    scored hits by distance here to get a genuinely most-relevant, stable top-k.
    """
    vector_search = NodeEdgeVectorSearch()
    await vector_search.embed_and_retrieve_distances(
        query=query,
        collections=list(_SEED_VECTOR_COLLECTIONS),
        wide_search_limit=wide_search_top_k,
    )
    if not vector_search.has_results():
        return []

    # Each collection's hits are ScoredResult(id, score); lower score = closer.
    scored = [
        (str(hit.id), hit.score)
        for hits in vector_search.node_distances.values()
        for hit in hits
        if getattr(hit, "id", None) is not None
    ]
    scored.sort(key=lambda pair: pair[1] if pair[1] is not None else float("inf"))
    return _unique_preserve_order([node_id for node_id, _ in scored])[:seed_top_k]


async def resolve_seeds_by_degree(graph_engine: Any, top_k: int) -> List[str]:
    """Highest-degree nodes as seeds — the default view when no seed is given.

    Uses ``get_graph_data()`` (implemented by every adapter) and counts degree
    in memory. This loads the whole graph, so it runs only as the no-seed
    fallback; query/ids/recall seeds expand via ``get_neighborhood`` without a
    full load.
    """
    nodes, edges = await graph_engine.get_graph_data()
    if not nodes:
        return []

    degree: Dict[str, int] = {str(node_id): 0 for node_id, _ in nodes}
    for edge in edges:
        source_key, target_key = str(edge[0]), str(edge[1])
        if source_key in degree:
            degree[source_key] += 1
        if target_key in degree:
            degree[target_key] += 1

    ranked = sorted(degree.items(), key=lambda item: item[1], reverse=True)
    return [node_id for node_id, _ in ranked[:top_k]]


async def resolve_seed_node_ids(
    graph_engine: Any,
    *,
    seed_node_ids: Optional[List[str]] = None,
    recall_result: Any = None,
    query: Optional[str] = None,
    seed_top_k: int = DEFAULT_SEED_TOP_K,
) -> Tuple[List[str], str]:
    """Resolve seeds by priority: explicit ids > recall > query > degree."""
    if seed_node_ids:
        return _unique_preserve_order([str(n) for n in seed_node_ids])[:seed_top_k], "explicit"

    recall_seeds = resolve_seeds_from_recall(recall_result)
    if recall_seeds:
        return recall_seeds[:seed_top_k], "recall"

    if query:
        query_seeds = await resolve_seeds_from_query(query, seed_top_k=seed_top_k)
        if query_seeds:
            return query_seeds, "query"

    degree_seeds = await resolve_seeds_by_degree(graph_engine, seed_top_k)
    return (degree_seeds, "degree") if degree_seeds else ([], "none")


def truncate_subgraph(
    nodes_data: List[Node],
    edges_data: List[EdgeData],
    seed_ids: List[str],
    max_nodes: int,
) -> Tuple[GraphData, bool]:
    """Cap the subgraph at ``max_nodes``, keeping seeds and their nearest nodes.

    Nodes are ranked by hop distance from the seeds (seeds first); edges are
    kept only when both endpoints survive, so no dangling edges remain.
    """
    if max_nodes <= 0 or len(nodes_data) <= max_nodes:
        return (nodes_data, edges_data), False

    adjacency: Dict[str, set] = {}
    for edge in edges_data:
        source_key, target_key = str(edge[0]), str(edge[1])
        adjacency.setdefault(source_key, set()).add(target_key)
        adjacency.setdefault(target_key, set()).add(source_key)

    hop_distance: Dict[str, int] = {}
    queue: deque = deque()
    for seed_id in seed_ids:
        seed_key = str(seed_id)
        if seed_key not in hop_distance:
            hop_distance[seed_key] = 0
            queue.append(seed_key)
    while queue:
        node_id = queue.popleft()
        for neighbor_id in adjacency.get(node_id, ()):
            if neighbor_id not in hop_distance:
                hop_distance[neighbor_id] = hop_distance[node_id] + 1
                queue.append(neighbor_id)

    node_rank = {str(node_id): index for index, (node_id, _) in enumerate(nodes_data)}
    kept_nodes = sorted(
        nodes_data,
        key=lambda item: (hop_distance.get(str(item[0]), 10_000), node_rank[str(item[0])]),
    )[:max_nodes]
    kept_ids = {str(node_id) for node_id, _ in kept_nodes}
    kept_edges = [
        edge for edge in edges_data if str(edge[0]) in kept_ids and str(edge[1]) in kept_ids
    ]
    return (kept_nodes, kept_edges), True


async def fetch_visualization_graph_data(
    graph_engine: Any,
    *,
    full: bool = False,
    query: Optional[str] = None,
    seed_node_ids: Optional[List[str]] = None,
    recall_result: Any = None,
    neighborhood_depth: int = DEFAULT_NEIGHBORHOOD_DEPTH,
    seed_top_k: int = DEFAULT_SEED_TOP_K,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> GraphData:
    """Return graph data for visualization: a bounded subgraph by default, or
    the whole graph when ``full`` is True. Output is the ``(nodes, edges)``
    format the renderer already consumes."""
    if neighborhood_depth < 1:
        raise ValueError("neighborhood_depth must be >= 1")
    if seed_top_k < 1:
        raise ValueError("seed_top_k must be >= 1")
    if max_nodes < 1:
        raise ValueError("max_nodes must be >= 1")

    if full:
        return await graph_engine.get_graph_data()

    seeds, source = await resolve_seed_node_ids(
        graph_engine,
        seed_node_ids=seed_node_ids,
        recall_result=recall_result,
        query=query,
        seed_top_k=seed_top_k,
    )
    if not seeds:
        logger.info("Subgraph visualization: no seeds resolved; rendering empty graph.")
        return ([], [])

    nodes_data, edges_data = await graph_engine.get_neighborhood(
        node_ids=seeds, depth=neighborhood_depth
    )
    (nodes_data, edges_data), truncated = truncate_subgraph(
        nodes_data, edges_data, seeds, max_nodes
    )
    logger.info(
        "Subgraph visualization: seeds=%d source=%s depth=%d nodes=%d edges=%d truncated=%s",
        len(seeds),
        source,
        neighborhood_depth,
        len(nodes_data),
        len(edges_data),
        truncated,
    )
    return (nodes_data, edges_data)
