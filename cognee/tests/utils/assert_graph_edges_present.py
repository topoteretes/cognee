from uuid import UUID
from typing import List, Tuple

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine.models.DataPoint import DataPoint


async def assert_graph_edges_present(relationships: List[Tuple[UUID, UUID, str]]):
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    nodes_by_id = {str(node[0]): node[1] for node in nodes}

    edge_ids = set([f"{str(edge[0])}_{edge[2]}_{str(edge[1])}" for edge in edges])

    for relationship in relationships:
        relationship_id = f"{str(relationship[0])}_{relationship[2]}_{str(relationship[1])}"
        relationship_name = relationship[2]
        assert relationship_id in edge_ids, (
            f"Edge '{relationship_name}' not present between '{nodes_by_id[str(relationship[0])]['name']}' and '{nodes_by_id[str(relationship[1])]['name']}' in graph database."
        )
