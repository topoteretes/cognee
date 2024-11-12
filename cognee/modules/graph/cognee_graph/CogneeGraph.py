from typing import List, Dict, Union
from CogneeGraphElements import Node, Edge
from CogneeAbstractGraph import CogneeAbstractGraph
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
from cognee.infrastructure.databases.graph.networkx.adapter import NetworkXAdapter
import os

class CogneeGraph(CogneeAbstractGraph):
    """
    Concrete implementation of the AbstractGraph class for Cognee.

    This class provides the functionality to manage nodes and edges,
    and project a graph from a database using adapters.
    """
    def add_node(self, node: Node) -> None:
        if node.id not in self.nodes:
            self.nodes[node.id] = node
        else:
            raise ValueError(f"Node with id {node.id} already exists.")

    # :TODO ADD dimension
    def add_edge(self, edge: Edge) -> None:
        if edge not in self.edges:
            self.edges.append(edge)
            edge.node1.add_skeleton_edge(edge)
            edge.node2.add_skeleton_edge(edge)
        else:
            raise ValueError(f"Edge {edge} already exists in the graph.")

    def get_node(self, node_id: str) -> Node:
        return self.nodes.get(node_id, None)

    def get_edges(self, node_id: str) -> List[Edge]:
        node = self.get_node(node_id)
        if node:
            return node.skeleton_edges
        else:
            raise ValueError(f"Node with id {node_id} does not exist.")

    # :TODO This should take also the list of entity types and connection types to keep. (Maybe we dont need all and can keep just an abstraction of the db network)
    async def project_graph_from_db(self, adapter: Union[Neo4jAdapter, NetworkXAdapter]) -> None:

        # :TODO: Handle networkx and Neo4j separately
        nodes_data, edges_data = await adapter.get_graph_data()

        raise NotImplementedError("To be implemented...tomorrow")


"""
The following code only used for test purposes and will be deleted later
"""
import asyncio

async def main():
    # Choose the adapter (Neo4j or NetworkX)
    adapter = await get_graph_engine()

    # Create an instance of CogneeGraph
    graph = CogneeGraph()

    # Project the graph from the database
    await graph.project_graph_from_db(adapter)

    # Access nodes and edges
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    print("Sample node:", graph.get_node("node1"))
    print("Edges for node1:", graph.get_edges("node1"))

# Run the main function
asyncio.run(main())