from typing import List

async def add_cognitive_layers(graph_client, parent_node_id: str, cognitive_layers: List):
    cognitive_layer_nodes = [(
        generate_cognitive_layer_id(cognitive_layer.name),
        dict(
            id = generate_cognitive_layer_id(cognitive_layer.name),
            name = cognitive_layer.name,
            description = cognitive_layer.description,
        ),
    ) for cognitive_layer in cognitive_layers]

    await graph_client.add_nodes(cognitive_layer_nodes)
    await graph_client.add_edges((
        parent_node_id,
        cognitive_layer_id,
        "decomposed_as",
        dict(relationship_name = "decomposed_as"),
    ) for (cognitive_layer_id, _) in cognitive_layer_nodes)

    return cognitive_layer_nodes

def generate_cognitive_layer_id(layer_id: str) -> str:
    return layer_id.upper().replace(" ", "_").replace("'", "").replace("/", "_")
