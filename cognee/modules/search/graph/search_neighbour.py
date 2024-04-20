""" Fetches the context of a given node in the graph"""
from typing import Union, Dict

from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
import networkx as nx
from cognee.shared.data_models import GraphDBType

async def search_neighbour(graph: Union[nx.Graph, any], id: str, infrastructure_config: Dict,
                           other_param: dict = None):
    """
    Search for nodes that share the same 'layer_uuid' as the specified node and return their descriptions.
    Adapts to both NetworkX graphs and Neo4j graph databases based on the configuration.

    Parameters:
    - graph (Union[nx.Graph, AsyncSession]): The graph object or Neo4j session.
    - id (str): The identifier of the node to match against.
    - infrastructure_config (Dict): Configuration that includes the graph engine type.
    - other_param (dict, optional): A dictionary that may contain 'node_id' to specify the node.

    Returns:
    - List[str]: A list of 'description' attributes of nodes that share the same 'layer_uuid' with the specified node.
    """
    node_id = other_param.get('node_id') if other_param else None

    if node_id is None:
        return []

    if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:
        if isinstance(graph, nx.Graph):
            if node_id not in graph:
                return []

            relevant_context = []
            target_layer_uuid = graph.nodes[node_id].get('layer_uuid')

            for n, attr in graph.nodes(data=True):
                if attr.get('layer_uuid') == target_layer_uuid and 'description' in attr:
                    relevant_context.append(attr['description'])

            return relevant_context
        else:
            raise ValueError("Graph object does not match the specified graph engine type in the configuration.")

    elif infrastructure_config.get_config()["graph_engine"] == GraphDBType.NEO4J:
        if isinstance(graph, AsyncSession):
            cypher_query = """
            MATCH (target {id: $node_id})
            WITH target.layer_uuid AS layer
            MATCH (n)
            WHERE n.layer_uuid = layer AND EXISTS(n.description)
            RETURN n.description AS description
            """
            result = await graph.run(cypher_query, node_id=node_id)
            descriptions = [record["description"] for record in await result.list()]

            return descriptions
        else:
            raise ValueError("Graph session does not match the specified graph engine type in the configuration.")

    else:
        raise ValueError("Unsupported graph engine type in the configuration.")



# if __name__ == '__main__':
#     import asyncio
#     async def main():
#         from cognee.shared.data_models import GraphDBType
#
#         graph_client = get_graph_client(GraphDBType.NETWORKX)
#         graph = await  graph_client.graph
#
#         await fetch_context(graph, "1")
#
#     asyncio.run(main())


