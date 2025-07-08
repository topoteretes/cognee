from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


async def get_edge_density(adapter: Neo4jAdapter):
    """
    Calculate the edge density of a graph in a Neo4j database.

    This function executes a Cypher query to determine the ratio of edges to the maximum
    possible edges in a graph, based on the number of nodes. If there are fewer than two
    nodes, it returns an edge density of zero.

    Parameters:
    -----------

        - adapter (Neo4jAdapter): An instance of Neo4jAdapter used to interface with the
          Neo4j database.

    Returns:
    --------

        Returns the calculated edge density as a float, or 0 if no results are found.
    """
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
    """
    Retrieve the number of connected components in a specified graph using the Neo4j
    adapter.

    Parameters:
    -----------

        - adapter (Neo4jAdapter): An instance of Neo4jAdapter for executing database
          queries.
        - graph_name (str): The name of the graph to analyze for connected components.

    Returns:
    --------

        Returns the number of connected components in the graph. Returns 0 if no results are
        found.
    """
    query = f"""
    CALL gds.wcc.stats('{graph_name}')
    YIELD componentCount
    RETURN componentCount AS num_connected_components;
    """

    result = await adapter.query(query)
    return result[0]["num_connected_components"] if result else 0


async def get_size_of_connected_components(adapter: Neo4jAdapter, graph_name: str):
    """
    Retrieve sizes of connected components in a graph.

    This function executes a query to calculate the sizes of connected components in the
    specified graph using the Graph Data Science library, and returns a list of these sizes
    in descending order.

    Parameters:
    -----------

        - adapter (Neo4jAdapter): An instance of Neo4jAdapter used to execute the database
          query.
        - graph_name (str): The name of the graph for which to retrieve connected component
          sizes.

    Returns:
    --------

        - list: A list of sizes of the connected components, ordered from largest to
          smallest. Returns an empty list if no results are found.
    """
    query = f"""
    CALL gds.wcc.stream('{graph_name}')
    YIELD componentId
    RETURN componentId, count(*) AS size
    ORDER BY size DESC;
    """

    result = await adapter.query(query)
    return [record["size"] for record in result] if result else []


async def count_self_loops(adapter: Neo4jAdapter):
    """
    Count the number of self-loop relationships in the Neo4j database.

    This function executes a Cypher query to find and count all edge relationships that
    begin and end at the same node (self-loops). It returns the count of such relationships
    or 0 if no results are found.

    Parameters:
    -----------

        - adapter (Neo4jAdapter): An instance of Neo4jAdapter used to interact with the
          Neo4j database.

    Returns:
    --------

        The count of self-loop relationships found in the database, or 0 if none were found.
    """
    query = """
    MATCH (n)-[r]->(n)
    RETURN count(r) AS adapter_loop_count;
    """
    result = await adapter.query(query)
    return result[0]["adapter_loop_count"] if result else 0


async def get_shortest_path_lengths(adapter: Neo4jAdapter, graph_name: str):
    """
    Fetches the shortest path lengths for a specified graph.

    Executes a Cypher query to retrieve the shortest path distances from a Neo4j graph
    represented by the given graph name. If no results are returned, an empty list is
    provided as output.

    Parameters:
    -----------

        - adapter (Neo4jAdapter): The Neo4jAdapter instance used to communicate with the
          Neo4j database.
        - graph_name (str): The name of the graph for which the shortest path lengths are to
          be retrieved.

    Returns:
    --------

        A list containing the shortest path distances or an empty list if no results are
        found.
    """
    query = f"""
    CALL gds.allShortestPaths.stream('{graph_name}')
    YIELD distance
    RETURN distance;
    """

    result = await adapter.query(query)
    return [res["distance"] for res in result] if result else []


async def get_avg_clustering(adapter: Neo4jAdapter, graph_name: str):
    """
    Calculate the average clustering coefficient for the specified graph.

    This function constructs a Cypher query to calculate the average of local clustering
    coefficients for all nodes in the provided graph. It utilizes the Neo4j Graph Data
    Science (GDS) library to execute the query asynchronously and return the computed
    average value.

    Parameters:
    -----------

        - adapter (Neo4jAdapter): An instance of Neo4jAdapter used to execute the query
          against the Neo4j database.
        - graph_name (str): The name of the graph for which the average clustering
          coefficient is to be calculated.

    Returns:
    --------

        The average clustering coefficient as a float, or 0 if no results are available.
    """
    query = f"""
    CALL gds.localClusteringCoefficient.stats('{graph_name}')
    YIELD averageClusteringCoefficient
    RETURN averageClusteringCoefficient AS avg_clustering;
    """

    result = await adapter.query(query)
    return result[0]["avg_clustering"] if result else 0
