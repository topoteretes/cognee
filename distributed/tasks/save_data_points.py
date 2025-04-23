import asyncio

from cognee.modules.graph.utils import deduplicate_nodes_and_edges, get_graph_from_model
from distributed.queues import save_data_points_queue


async def save_data_points(
    data_points_and_relationships: tuple[list, list]
):
    data_points = data_points_and_relationships[0]
    data_point_connections = data_points_and_relationships[1]

    nodes = []
    edges = []

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    results = await asyncio.gather(
        *[
            get_graph_from_model(
                data_point,
                added_nodes=added_nodes,
                added_edges=added_edges,
                visited_properties=visited_properties,
            )
            for data_point in data_points
        ]
    )

    for result_nodes, result_edges in results:
        nodes.extend(result_nodes)
        edges.extend(result_edges)

    nodes, edges = deduplicate_nodes_and_edges(nodes, edges + data_point_connections)

    # await index_data_points(nodes)

    save_data_points_queue.put((nodes, edges))
