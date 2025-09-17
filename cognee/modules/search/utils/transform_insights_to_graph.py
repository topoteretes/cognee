from typing import Dict, List, Tuple


def transform_insights_to_graph(context: List[Tuple[Dict, Dict, Dict]]):
    nodes = {}
    edges = {}

    for triplet in context:
        nodes[triplet[0]["id"]] = {
            "id": triplet[0]["id"],
            "label": triplet[0]["name"] if "name" in triplet[0] else triplet[0]["id"],
            "type": triplet[0]["type"],
        }
        nodes[triplet[2]["id"]] = {
            "id": triplet[2]["id"],
            "label": triplet[2]["name"] if "name" in triplet[2] else triplet[2]["id"],
            "type": triplet[2]["type"],
        }
        edges[f"{triplet[0]['id']}_{triplet[1]['relationship_name']}_{triplet[2]['id']}"] = {
            "source": triplet[0]["id"],
            "target": triplet[2]["id"],
            "label": triplet[1]["relationship_name"],
        }

    return {
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }
