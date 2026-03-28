"""Serialize search result objects, breaking circular Edge↔Node references."""

from typing import Any, List, Optional, Union

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node


def _serialize_node(node: Node) -> dict:
    """Convert a Node to a JSON-serializable dict without its skeleton_edges."""
    return {
        "id": node.id,
        "attributes": node.attributes,
    }


def _serialize_edge(edge: Edge) -> dict:
    """Convert an Edge to a JSON-serializable dict, breaking circular refs."""
    return {
        "source_node_id": edge.node1.id,
        "target_node_id": edge.node2.id,
        "source_node_attributes": edge.node1.attributes,
        "target_node_attributes": edge.node2.attributes,
        "edge_attributes": edge.attributes,
        "directed": edge.directed,
    }


def serialize_result_objects(
    objects: Optional[Union[List[Any], Any]],
) -> Optional[Union[List[Any], Any]]:
    """Serialize result objects, converting Edge/Node instances to plain dicts.

    This prevents ``jsonable_encoder`` from hitting infinite recursion caused by
    the circular reference chain: Edge.node1 → Node.skeleton_edges → [Edge, …].
    """
    if objects is None:
        return None

    if isinstance(objects, list):
        return [_serialize_single(obj) for obj in objects]

    return _serialize_single(objects)


def _serialize_single(obj: Any) -> Any:
    if isinstance(obj, Edge):
        return _serialize_edge(obj)
    if isinstance(obj, Node):
        return _serialize_node(obj)
    return obj
