from cognee.infrastructure.engine import DataPoint


def deduplicate_nodes_and_edges(nodes: list[DataPoint], edges: list[dict]):
    added_entities = {}
    final_nodes = []
    final_edges = []

    for node in nodes:
        if str(node.id) not in added_entities:
            final_nodes.append(node)
            added_entities[str(node.id)] = True

    for edge in edges:
        edge_key = str(edge[0]) + str(edge[2]) + str(edge[1])
        if edge_key not in added_entities:
            final_edges.append(edge)
            added_entities[edge_key] = True

    return final_nodes, final_edges
