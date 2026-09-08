from typing import Dict, List, Tuple
from uuid import UUID

from cognee.infrastructure.databases.graph import get_graph_engine


async def assert_graph_edges_not_present(relationships: list[tuple[UUID, UUID, str, dict]]):
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    nodes_by_id = {str(node[0]): node[1] for node in nodes}

    edge_ids = {f"{edge[0]!s}_{edge[2]}_{edge[1]!s}" for edge in edges}

    for relationship in relationships:
        relationship_id = f"{relationship[0]!s}_{relationship[2]}_{relationship[1]!s}"

        if relationship_id in edge_ids:
            relationship_name = relationship[2]
            source_node = nodes_by_id[str(relationship[0])]
            destination_node = nodes_by_id[str(relationship[1])]
            assert False, (
                f"Edge '{relationship_name}' still present between '{source_node['name'] if 'node' in source_node else source_node['id']}' and '{destination_node['name'] if 'node' in destination_node else destination_node['id']}' in graph database."
            )
