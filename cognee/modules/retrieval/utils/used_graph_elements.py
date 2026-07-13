"""Helpers to extract used graph element IDs from retriever results for session QA."""

from typing import Any, Dict, List, Optional, Union

from cognee.modules.search.types.SearchType import SearchType

SUBGRAPH_SEARCH_TYPES = frozenset(
    {
        SearchType.GRAPH_COMPLETION,
        SearchType.GRAPH_COMPLETION_DECOMPOSITION,
        SearchType.GRAPH_COMPLETION_COT,
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        SearchType.GRAPH_SUMMARY_COMPLETION,
        SearchType.TRIPLET_COMPLETION,
        SearchType.TEMPORAL,
        SearchType.AGENTIC_COMPLETION,
    }
)

_EMPTY_SUBGRAPH: Dict[str, List] = {"nodes": [], "edges": []}


def is_edge_list(obj: Any) -> bool:
    """Duck-type: list of objects with node1, node2, attributes (CogneeGraph Edge)."""
    if not isinstance(obj, list) or not obj:
        return False
    first = obj[0]
    return (
        getattr(first, "node1", None) is not None
        and getattr(first, "node2", None) is not None
        and getattr(first, "attributes", None) is not None
    )


def extract_from_edges(edges: List[Any]) -> Optional[Dict[str, List[str]]]:
    """
    From a list of Edge-like objects, collect node_ids and edge_ids.
    edge_ids only from attributes["edge_object_id"] when present (no fallback).
    Returns None if nothing extracted.
    """
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    for edge in edges:
        try:
            n1 = getattr(edge, "node1", None)
            n2 = getattr(edge, "node2", None)
            attrs = getattr(edge, "attributes", None) or {}
            if n1 is not None and getattr(n1, "id", None) is not None:
                node_ids.add(str(n1.id))
            if n2 is not None and getattr(n2, "id", None) is not None:
                node_ids.add(str(n2.id))
            eid = attrs.get("edge_object_id")
            if eid is not None:
                edge_ids.add(str(eid))
        except (TypeError, AttributeError):
            continue
    result: Dict[str, List[str]] = {}
    if node_ids:
        result["node_ids"] = sorted(node_ids)
    if edge_ids:
        result["edge_ids"] = sorted(edge_ids)
    return result if result else None


def extract_from_scored_results(results: List[Any]) -> Optional[Dict[str, List[str]]]:
    """
    From ScoredResult-like list: node_ids from payload["id"] if present, else .id.
    No edge_ids. Returns None if nothing extracted.
    """
    node_ids: set[str] = set()
    for r in results:
        try:
            pid = None
            if getattr(r, "payload", None) and isinstance(r.payload, dict):
                pid = r.payload.get("id")
            if pid is not None:
                node_ids.add(str(pid))
            elif getattr(r, "id", None) is not None:
                node_ids.add(str(r.id))
        except (TypeError, AttributeError):
            continue
    if not node_ids:
        return None
    return {"node_ids": sorted(node_ids)}


def extract_from_temporal_dict(obj: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
    """
    From temporal retriever dict: triplets -> extract_from_edges; events path -> extract_from_scored_results.
    Returns None if nothing extracted.
    """
    if not isinstance(obj, dict):
        return None
    triplets = obj.get("triplets")
    if triplets is not None and is_edge_list(triplets):
        return extract_from_edges(triplets)
    scored_results = obj.get("vector_search_results")
    if isinstance(scored_results, list) and scored_results:
        return extract_from_scored_results(scored_results)
    return None


def supports_subgraph_search_type(query_type: SearchType) -> bool:
    """Return True when the search type can expose a retrieved subgraph stub."""
    return query_type in SUBGRAPH_SEARCH_TYPES


def _node_display_name(node: Any) -> str:
    attributes = getattr(node, "attributes", None) or {}
    name = attributes.get("name")
    if name:
        return str(name)
    text = attributes.get("text")
    if text:
        return str(text)[:120]
    description = attributes.get("description")
    if description:
        return str(description)[:120]
    return "Unnamed Node"


def _node_type_label(node: Any) -> str:
    attributes = getattr(node, "attributes", None) or {}
    return str(attributes.get("type") or attributes.get("node_type") or "Entity")


def _edge_relationship(edge: Any) -> str:
    attributes = getattr(edge, "attributes", None) or {}
    return str(
        attributes.get("relationship_type")
        or attributes.get("relationship_name")
        or attributes.get("edge_text")
        or ""
    )


def _vector_distance_score(element: Any, query_index: int = 0) -> Optional[float]:
    attributes = getattr(element, "attributes", None) or {}
    distances = attributes.get("vector_distance")
    if not isinstance(distances, list) or query_index >= len(distances):
        return None
    try:
        return float(distances[query_index])
    except (TypeError, ValueError):
        return None


def _scored_result_score(result: Any) -> Optional[float]:
    score = getattr(result, "score", None)
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _node_stub(node: Any) -> Dict[str, str]:
    return {
        "id": str(node.id),
        "type": _node_type_label(node),
        "name": _node_display_name(node),
    }


def build_subgraph_stub_from_edges(
    edges: List[Any], query_index: int = 0
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a compact subgraph stub from retrieved Edge objects."""
    nodes_by_id: Dict[str, Dict[str, str]] = {}
    edge_stubs: List[Dict[str, Any]] = []

    for edge in edges:
        try:
            node1 = getattr(edge, "node1", None)
            node2 = getattr(edge, "node2", None)
            if node1 is None or node2 is None:
                continue
            if getattr(node1, "id", None) is not None:
                nodes_by_id.setdefault(str(node1.id), _node_stub(node1))
            if getattr(node2, "id", None) is not None:
                nodes_by_id.setdefault(str(node2.id), _node_stub(node2))

            edge_stub: Dict[str, Any] = {
                "source": str(node1.id),
                "target": str(node2.id),
                "relationship": _edge_relationship(edge),
            }
            score = _vector_distance_score(edge, query_index)
            if score is not None:
                edge_stub["score"] = score
            edge_stubs.append(edge_stub)
        except (TypeError, AttributeError):
            continue

    return {"nodes": list(nodes_by_id.values()), "edges": edge_stubs}


def build_subgraph_stub_from_scored_results(
    results: List[Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a nodes-only subgraph stub from scored vector results (e.g. temporal events)."""
    nodes_by_id: Dict[str, Dict[str, Any]] = {}

    for result in results:
        try:
            payload = getattr(result, "payload", None)
            if not isinstance(payload, dict):
                payload = {}
            node_id = payload.get("id") or getattr(result, "id", None)
            if node_id is None:
                continue
            node_id = str(node_id)
            if node_id in nodes_by_id:
                continue
            node_stub: Dict[str, Any] = {
                "id": node_id,
                "type": str(payload.get("type") or payload.get("node_type") or "Entity"),
                "name": str(
                    payload.get("name")
                    or payload.get("description")
                    or payload.get("text")
                    or "Unnamed Node"
                ),
            }
            score = _scored_result_score(result)
            if score is not None:
                node_stub["score"] = score
            nodes_by_id[node_id] = node_stub
        except (TypeError, AttributeError):
            continue

    return {"nodes": list(nodes_by_id.values()), "edges": []}


def _is_batch_edge_lists(obj: Any) -> bool:
    return (
        isinstance(obj, list)
        and bool(obj)
        and isinstance(obj[0], list)
        and (not obj[0] or is_edge_list(obj[0]))
    )


def build_retrieved_subgraph(
    retrieved_objects: Any,
    query_type: SearchType,
    query_index: int = 0,
) -> Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, List[Dict[str, Any]]]]]:
    """Build the API subgraph stub from in-memory retriever output."""
    if not supports_subgraph_search_type(query_type):
        return None  # type: ignore[return-value]

    if retrieved_objects is None:
        return dict(_EMPTY_SUBGRAPH)

    if isinstance(retrieved_objects, dict):
        triplets = retrieved_objects.get("triplets")
        if triplets is not None:
            if _is_batch_edge_lists(triplets):
                return [
                    build_subgraph_stub_from_edges(batch, query_index=query_index)
                    for batch in triplets
                ]
            if is_edge_list(triplets):
                return build_subgraph_stub_from_edges(triplets, query_index=query_index)
            if isinstance(triplets, list) and not triplets:
                return dict(_EMPTY_SUBGRAPH)

        vector_search_results = retrieved_objects.get("vector_search_results")
        if isinstance(vector_search_results, list):
            if not vector_search_results:
                return dict(_EMPTY_SUBGRAPH)
            return build_subgraph_stub_from_scored_results(vector_search_results)

        return dict(_EMPTY_SUBGRAPH)

    if _is_batch_edge_lists(retrieved_objects):
        return [
            build_subgraph_stub_from_edges(batch, query_index=query_index)
            for batch in retrieved_objects
        ]

    if is_edge_list(retrieved_objects):
        return build_subgraph_stub_from_edges(retrieved_objects, query_index=query_index)

    if isinstance(retrieved_objects, list) and not retrieved_objects:
        return dict(_EMPTY_SUBGRAPH)

    return dict(_EMPTY_SUBGRAPH)
