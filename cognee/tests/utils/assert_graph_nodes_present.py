from typing import List
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine.models.DataPoint import DataPoint


async def assert_graph_nodes_present(data_points: List[DataPoint]):
    graph_engine = await get_graph_engine()
    nodes, __ = await graph_engine.get_graph_data()

    node_ids = set(node[0] for node in nodes)

    for data_point in data_points:
        node_name = getattr(data_point, "label", getattr(data_point, "name", data_point.id))
        assert str(data_point.id) in node_ids, f"Node '{node_name}' not found in graph database."
