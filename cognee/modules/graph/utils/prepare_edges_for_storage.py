"""Prepare edges for graph storage by ensuring each has a stable edge_object_id."""

from typing import Any, Dict, List, Tuple

from cognee.modules.engine.utils import generate_edge_object_id


def ensure_edge_object_ids(
    edges: List[Tuple[str, str, str, Dict[str, Any]]],
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Ensure each edge has edge_object_id in its properties dict.
    Returns a new list of edges; does not mutate the input.
    """
    result = []
    for source_id, target_id, relationship_name, properties in edges:
        props = dict(properties) if properties else {}
        if "edge_object_id" not in props:
            props["edge_object_id"] = generate_edge_object_id(
                source_id, target_id, relationship_name
            )
        result.append((source_id, target_id, relationship_name, props))
    return result
