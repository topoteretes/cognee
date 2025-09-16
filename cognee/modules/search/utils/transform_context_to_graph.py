from typing import List

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


def transform_context_to_graph(context: List[Edge]):
    nodes = {}
    edges = {}

    for triplet in context:
        nodes[triplet.node1.id] = {
            "id": triplet.node1.id,
            "label": triplet.node1.attributes["name"]
            if "name" in triplet.node1.attributes
            else triplet.node1.id,
            "type": triplet.node1.attributes["type"],
            "attributes": triplet.node1.attributes,
        }
        nodes[triplet.node2.id] = {
            "id": triplet.node2.id,
            "label": triplet.node2.attributes["name"]
            if "name" in triplet.node2.attributes
            else triplet.node2.id,
            "type": triplet.node2.attributes["type"],
            "attributes": triplet.node2.attributes,
        }
        edges[
            f"{triplet.node1.id}_{triplet.attributes['relationship_name']}_{triplet.node2.id}"
        ] = {
            "source": triplet.node1.id,
            "target": triplet.node2.id,
            "label": triplet.attributes["relationship_name"],
        }

    return {
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }
