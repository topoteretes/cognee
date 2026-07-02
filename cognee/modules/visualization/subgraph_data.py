"""Bounded subgraph selection for graph visualization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch
from cognee.modules.retrieval.utils.used_graph_elements import (
    extract_from_edges,
    extract_from_temporal_dict,
    is_edge_list,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("visualization.subgraph_data")

DEFAULT_NEIGHBORHOOD_DEPTH = 2
DEFAULT_SEED_TOP_K = 10
DEFAULT_MAX_NODES = 500
DEFAULT_WIDE_SEARCH_TOP_K = 100

_DEFAULT_VECTOR_COLLECTIONS = [
    "Entity_name",
    "TextSummary_text",
    "EntityType_name",
    "DocumentChunk_text",
    "EdgeType_relationship_name",
]

NodeData = Tuple[str, Dict[str, Any]]
EdgeData = Tuple[str, str, str, Dict[str, Any]]
GraphData = Tuple[List[NodeData], List[EdgeData]]


def _unique_preserve_order(node_ids: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for node_id in node_ids:
        key = str(node_id)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


@dataclass(frozen=True)
class SubgraphMetadata:
    scope: Literal["subgraph", "all"]
    seed_ids: List[str]
    depth: int
    max_nodes: int
    seed_top_k: int
    truncated: bool
    seed_source: str


def resolve_seeds_from_recall(recall_result: Any) -> List[str]:
    """Extract seed node IDs from recall/search payloads."""
    if recall_result is None:
        return []

    if isinstance(recall_result, dict):
        if recall_result.get("node_ids"):
            return _unique_preserve_order([str(node_id) for node_id in recall_result["node_ids"]])
        extracted = extract_from_temporal_dict(recall_result)
        if extracted and extracted.get("node_ids"):
            return list(extracted["node_ids"])

    if is_edge_list(recall_result):
        extracted = extract_from_edges(recall_result)
        return list(extracted["node_ids"]) if extracted and extracted.get("node_ids") else []

    items = recall_result if isinstance(recall_result, list) else [recall_result]
    node_ids: List[str] = []
    for item in items:
        used = getattr(item, "used_graph_element_ids", None)
        if isinstance(used, dict) and used.get("node_ids"):
            node_ids.extend(str(node_id) for node_id in used["node_ids"])
            continue

        metadata = getattr(item, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("node_ids"):
            node_ids.extend(str(node_id) for node_id in metadata["node_ids"])
            continue

        raw = getattr(item, "raw", None)
        if isinstance(raw, dict):
            if raw.get("node_ids"):
                node_ids.extend(str(node_id) for node_id in raw["node_ids"])
                continue
            extracted = extract_from_temporal_dict(raw)
            if extracted and extracted.get("node_ids"):
                node_ids.extend(extracted["node_ids"])

        if is_edge_list(getattr(item, "result_object", None)):
            extracted = extract_from_edges(item.result_object)
            if extracted and extracted.get("node_ids"):
                node_ids.extend(extracted["node_ids"])

    # Deduplicate while preserving order
    return _unique_preserve_order(node_ids)


async def resolve_seeds_from_query(
    query: str,
    seed_top_k: int = DEFAULT_SEED_TOP_K,
    wide_search_top_k: int = DEFAULT_WIDE_SEARCH_TOP_K,
) -> List[str]:
    """Resolve seed node IDs from a query via vector search."""
    vector_search = NodeEdgeVectorSearch()
    await vector_search.embed_and_retrieve_distances(
        query=query,
        collections=list(_DEFAULT_VECTOR_COLLECTIONS),
        wide_search_limit=wide_search_top_k,
    )
    if not vector_search.has_results():
        return []
    return vector_search.extract_relevant_node_ids()[:seed_top_k]


async def resolve_seeds_by_degree(graph_engine: Any, top_k: int) -> List[str]:
    """Pick top-degree nodes as seeds (Cypher when available, else in-memory)."""
    try:
        rows = await graph_engine.query(
            """
            MATCH (n:Node)-[r]-()
            WITH n, count(r) AS degree
            ORDER BY degree DESC
            LIMIT $limit
            RETURN n.id
            """,
            {"limit": top_k},
        )
        if rows:
            return [str(row[0]) for row in rows if row and row[0]][:top_k]
    except (NotImplementedError, Exception) as error:
        logger.debug("Degree query unavailable (%s); computing from graph data.", error)

    nodes, edges = await graph_engine.get_graph_data()
    if not nodes:
        return []

    degree: Dict[str, int] = {str(node_id): 0 for node_id, _ in nodes}
    for source_id, target_id, _, _ in edges:
        source_key = str(source_id)
        target_key = str(target_id)
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
    user: Any = None,
    session_ids: Optional[List[str]] = None,
) -> Tuple[List[str], str]:
    """Resolve seeds using priority: explicit > recall > query > session > degree."""
    if seed_node_ids:
        return [str(node_id) for node_id in seed_node_ids[:seed_top_k]], "explicit"

    recall_seeds = resolve_seeds_from_recall(recall_result)
    if recall_seeds:
        return recall_seeds[:seed_top_k], "recall"

    if query:
        query_seeds = await resolve_seeds_from_query(query, seed_top_k=seed_top_k)
        if query_seeds:
            return query_seeds, "query"

    from cognee.modules.visualization.session_events import get_latest_session_seed_node_ids

    session_seeds = await get_latest_session_seed_node_ids(user=user, session_ids=session_ids)
    if session_seeds:
        return session_seeds[:seed_top_k], "session"

    degree_seeds = await resolve_seeds_by_degree(graph_engine, seed_top_k)
    if degree_seeds:
        return degree_seeds, "degree"

    return [], "none"


def truncate_subgraph(
    nodes_data: List[NodeData],
    edges_data: List[EdgeData],
    seed_ids: List[str],
    max_nodes: int,
) -> Tuple[GraphData, bool]:
    """Cap node count by BFS hop distance from seeds (seeds always kept)."""
    if max_nodes <= 0 or len(nodes_data) <= max_nodes:
        return (nodes_data, edges_data), False

    adjacency: Dict[str, set[str]] = {}
    for source_id, target_id, _, _ in edges_data:
        source_key = str(source_id)
        target_key = str(target_id)
        adjacency.setdefault(source_key, set()).add(target_key)
        adjacency.setdefault(target_key, set()).add(source_key)

    hop_distance: Dict[str, int] = {}
    queue: deque[Tuple[str, int]] = deque()
    for seed_id in seed_ids:
        seed_key = str(seed_id)
        if seed_key not in hop_distance:
            hop_distance[seed_key] = 0
            queue.append((seed_key, 0))

    while queue:
        node_id, distance = queue.popleft()
        for neighbor_id in adjacency.get(node_id, ()):
            if neighbor_id not in hop_distance:
                hop_distance[neighbor_id] = distance + 1
                queue.append((neighbor_id, distance + 1))

    node_rank = {str(node_id): idx for idx, (node_id, _) in enumerate(nodes_data)}
    ranked_nodes = sorted(
        nodes_data,
        key=lambda item: (
            hop_distance.get(str(item[0]), 10_000),
            node_rank.get(str(item[0]), 10_000),
        ),
    )
    kept_nodes = ranked_nodes[:max_nodes]
    kept_ids = {str(node_id) for node_id, _ in kept_nodes}
    kept_edges = [
        edge for edge in edges_data if str(edge[0]) in kept_ids and str(edge[1]) in kept_ids
    ]
    return (kept_nodes, kept_edges), len(nodes_data) > max_nodes


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
    user: Any = None,
    session_ids: Optional[List[str]] = None,
) -> Tuple[GraphData, SubgraphMetadata]:
    """Return graph data for visualization (bounded subgraph by default)."""
    if neighborhood_depth < 1:
        raise ValueError("neighborhood_depth must be >= 1")
    if seed_top_k < 1:
        raise ValueError("neighborhood_seed_top_k must be >= 1")
    if max_nodes < 1:
        raise ValueError("max_nodes must be >= 1")

    if full:
        nodes_data, edges_data = await graph_engine.get_graph_data()
        return (nodes_data, edges_data), SubgraphMetadata(
            scope="all",
            seed_ids=[],
            depth=neighborhood_depth,
            max_nodes=max_nodes,
            seed_top_k=seed_top_k,
            truncated=False,
            seed_source="full",
        )

    seeds, seed_source = await resolve_seed_node_ids(
        graph_engine,
        seed_node_ids=seed_node_ids,
        recall_result=recall_result,
        query=query,
        seed_top_k=seed_top_k,
        user=user,
        session_ids=session_ids,
    )

    if not seeds:
        logger.warning("No seed nodes resolved for subgraph visualization; rendering empty graph.")
        return ([], []), SubgraphMetadata(
            scope="subgraph",
            seed_ids=[],
            depth=neighborhood_depth,
            max_nodes=max_nodes,
            seed_top_k=seed_top_k,
            truncated=False,
            seed_source=seed_source,
        )

    nodes_data, edges_data = await graph_engine.get_neighborhood(
        node_ids=seeds,
        depth=neighborhood_depth,
    )

    if not nodes_data and not edges_data:
        logger.warning(
            "Neighborhood expansion returned empty for seeds %s at depth %d.",
            seeds,
            neighborhood_depth,
        )
        return ([], []), SubgraphMetadata(
            scope="subgraph",
            seed_ids=seeds,
            depth=neighborhood_depth,
            max_nodes=max_nodes,
            seed_top_k=seed_top_k,
            truncated=False,
            seed_source=seed_source,
        )

    graph_data, truncated = truncate_subgraph(nodes_data, edges_data, seeds, max_nodes)
    return graph_data, SubgraphMetadata(
        scope="subgraph",
        seed_ids=seeds,
        depth=neighborhood_depth,
        max_nodes=max_nodes,
        seed_top_k=seed_top_k,
        truncated=truncated,
        seed_source=seed_source,
    )
