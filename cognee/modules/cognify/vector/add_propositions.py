import asyncio

from cognee.infrastructure.databases.vector import DataPoint
from cognee.infrastructure import infrastructure_config

def convert_to_data_point(node):
    return DataPoint(
        id = node["node_id"],
        payload = {
            "text": node["description"]
        },
        embed_field = "text"
    )

async def add_propositions(nodes_by_layer):
    vector_engine = infrastructure_config.get_config()["vector_engine"]

    awaitables = []

    for layer_id, layer_nodes in nodes_by_layer.items():
        awaitables.append(
            vector_engine.create_data_points(
                collection_name = layer_id,
                data_points = list(map(convert_to_data_point, layer_nodes))
            )
        )

    return await asyncio.gather(*awaitables)
