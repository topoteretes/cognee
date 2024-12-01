import asyncio
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.utils import get_graph_from_model
from .index_data_points import index_data_points


async def add_data_points(data_points: list[DataPoint]):
    nodes = []
    edges = []

    results = await asyncio.gather(*[
        get_graph_from_model(data_point) for data_point in data_points
    ])

    for result_nodes, result_edges in results:
        nodes.extend(result_nodes)
        edges.extend(result_edges)

    graph_engine = await get_graph_engine()

    await index_data_points(data_points)

    await graph_engine.add_nodes(nodes)
    await graph_engine.add_edges(edges)

    return data_points
