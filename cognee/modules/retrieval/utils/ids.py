from typing import Any

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


def normalize_id(value: Any) -> str:
    """Return a stable string key for graph/vector identifiers."""
    if value is None:
        return ""
    return str(value)


def triplet_key(edge: Edge) -> tuple[str, str, bool, str]:
    """Return a value-based key for a retrieved graph triplet."""
    attributes = getattr(edge, "attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}

    node1 = getattr(edge, "node1", None)
    node2 = getattr(edge, "node2", None)
    source_id = normalize_id(getattr(node1, "id", None))
    target_id = normalize_id(getattr(node2, "id", None))
    if not source_id and not target_id and not attributes:
        return ("", "", bool(getattr(edge, "directed", True)), repr(edge))

    relationship_key = (
        attributes.get("edge_object_id")
        or attributes.get("edge_type_id")
        or attributes.get("relationship_name")
        or ""
    )
    return (
        source_id,
        target_id,
        bool(getattr(edge, "directed", True)),
        normalize_id(relationship_key),
    )
