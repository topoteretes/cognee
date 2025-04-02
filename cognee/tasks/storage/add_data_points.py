import asyncio
from typing import List
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.utils import deduplicate_nodes_and_edges, get_graph_from_model
from .index_data_points import index_data_points
from .index_graph_edges import index_graph_edges


async def add_data_points(data_points: List[DataPoint]) -> List[DataPoint]:
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

    nodes, edges = deduplicate_nodes_and_edges(nodes, edges)

    graph_engine = await get_graph_engine()

    await index_data_points(nodes)

    await graph_engine.add_nodes(nodes)
    await graph_engine.add_edges(edges)

    # This step has to happen after adding nodes and edges because we query the graph.
    await index_graph_edges()

    return data_points
