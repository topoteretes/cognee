"""Helpers to extract used graph element IDs from retriever results for session QA."""

from typing import Any, Dict, List, Optional


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
