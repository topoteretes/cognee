from cognee.infrastructure.databases.graph.deduplicate_edges import deduplicate_edges
from cognee.infrastructure.engine import DataPoint


def deduplicate_nodes_and_edges(nodes: list[DataPoint], edges: list[dict]):
    """Drop repeated nodes (by id) and edges (by source, target, relationship).

    First occurrence wins for both. The edge pass is the shared ``keep="first"``
    policy of :mod:`cognee.infrastructure.databases.graph.deduplicate_edges`;
    see that module for how it relates to the adapter-level collapse.
    """
    added_entities = {}
    final_nodes = []

    for node in nodes:
        if str(node.id) not in added_entities:
            final_nodes.append(node)
            added_entities[str(node.id)] = True

    final_edges = deduplicate_edges(edges, keep="first")

    return final_nodes, final_edges
