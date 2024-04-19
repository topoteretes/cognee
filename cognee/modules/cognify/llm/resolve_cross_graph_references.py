from typing import Dict, List
from cognee.infrastructure import infrastructure_config

async def resolve_cross_graph_references(nodes_by_layer: Dict):
    results = []

    unique_layers = nodes_by_layer.keys()

    for layer_id, layer_nodes in nodes_by_layer.items():
        # Filter unique_layer_uuids to exclude the current layer
        other_layers = [uuid for uuid in unique_layers if uuid != layer_id]

        for other_layer in other_layers:
            results.append(await get_nodes_by_layer(other_layer, layer_nodes))

    return results

async def get_nodes_by_layer(layer_id: str, layer_nodes: List):
    vector_engine = infrastructure_config.get_config()["vector_engine"]

    score_points = await vector_engine.batch_search(
        layer_id,
        [layer_node["description"] for layer_node in layer_nodes],
        limit = 2
    )

    return {
        "layer_id": layer_id,
        "layer_nodes": connect_score_points_to_node(score_points, layer_nodes)
    }

def connect_score_points_to_node(score_points, layer_nodes):
    return [
        {
            "id": node["node_id"],
            "score_points": score_points[node_index]
        } for node_index, node in enumerate(layer_nodes)
    ]
