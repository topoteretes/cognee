import logging

from cognee.infrastructure.engine import DataPoint

logger = logging.getLogger(__name__)

# Fields that change on every construction and must not trigger a contradiction warning.
_VOLATILE_FIELDS = frozenset({"created_at", "updated_at", "valid_to", "version", "id"})


def _content_differs(existing: DataPoint, incoming: DataPoint) -> bool:
    """Return True when two same-id DataPoints have different content fields."""
    ex_dict = existing.model_dump(exclude=_VOLATILE_FIELDS)
    in_dict = incoming.model_dump(exclude=_VOLATILE_FIELDS)
    return ex_dict != in_dict


def deduplicate_nodes_and_edges(nodes: list[DataPoint], edges: list[dict]):
    added_entities: dict = {}
    final_nodes: list = []
    final_edges: list = []

    for node in nodes:
        node_key = str(node.id)
        if node_key not in added_entities:
            final_nodes.append(node)
            added_entities[node_key] = node
        else:
            existing = added_entities[node_key]
            if _content_differs(existing, node):
                logger.warning(
                    "Contradiction detected for node %s (%s): "
                    "two versions with different content in the same batch. "
                    "Keeping the first; consider closing the old node with close_node().",
                    node_key,
                    type(node).__name__,
                )

    for edge in edges:
        edge_key = str(edge[0]) + str(edge[2]) + str(edge[1])
        if edge_key not in added_entities:
            final_edges.append(edge)
            added_entities[edge_key] = True

    return final_nodes, final_edges
