from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


async def get_edge_density(adapter: Neo4jAdapter):
    query = """
    MATCH (n)
    WITH count(n) AS num_nodes
    MATCH ()-[r]->()
    WITH num_nodes, count(r) AS num_edges
    RETURN CASE
        WHEN num_nodes < 2 THEN 0
        ELSE num_edges * 1.0 / (num_nodes * (num_nodes - 1))
    END AS edge_density;
    """
    result = await adapter.query(query)
    return result[0]["edge_density"] if result else 0


async def get_num_connected_components(adapter: Neo4jAdapter, graph_name: str):
    query = f"""
    CALL gds.wcc.stream('{graph_name}')
    YIELD componentId
    RETURN count(DISTINCT componentId) AS num_connected_components;
    """

    result = await adapter.query(query)
    return result[0]["num_connected_components"] if result else 0


async def get_size_of_connected_components(adapter: Neo4jAdapter, graph_name: str):
    query = f"""
    CALL gds.wcc.stream('{graph_name}')
    YIELD componentId
    RETURN componentId, count(*) AS size
    ORDER BY size DESC;
    """

    result = await adapter.query(query)
    return [record["size"] for record in result] if result else []


async def count_self_loops(adapter: Neo4jAdapter):
    query = """
    MATCH (n)-[r]->(n)
    RETURN count(r) AS adapter_loop_count;
    """
    result = await adapter.query(query)
    return result[0]["adapter_loop_count"] if result else 0


async def get_shortest_path_lengths(adapter: Neo4jAdapter, graph_name: str):
    query = f"""
    CALL gds.allShortestPaths.stream('{graph_name}')
    YIELD distance
    RETURN distance;
    """

    result = await adapter.query(query)
    return [res["distance"] for res in result] if result else []


async def get_avg_clustering(adapter: Neo4jAdapter, graph_name: str):
    query = f"""
    CALL gds.localClusteringCoefficient.stream('{graph_name}')
    YIELD localClusteringCoefficient
    RETURN avg(localClusteringCoefficient) AS avg_clustering;
    """

    result = await adapter.query(query)
    return result[0]["avg_clustering"] if result else 0
