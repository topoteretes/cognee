from typing import List, Dict, Union

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge
from cognee.modules.graph.cognee_graph.CogneeAbstractGraph import CogneeAbstractGraph
from cognee.infrastructure.databases.graph import get_graph_engine

class CogneeGraph(CogneeAbstractGraph):
    """
    Concrete implementation of the AbstractGraph class for Cognee.

    This class provides the functionality to manage nodes and edges,
    and project a graph from a database using adapters.
    """

    nodes: Dict[str, Node]
    edges: List[Edge]
    directed: bool

    def __init__(self, directed: bool = True):
        self.nodes = {}
        self.edges = []
        self.directed = directed

    def add_node(self, node: Node) -> None:
        if node.id not in self.nodes:
            self.nodes[node.id] = node
        else:
            raise ValueError(f"Node with id {node.id} already exists.")

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

    async def project_graph_from_db(self,
                                    adapter: Union[GraphDBInterface],
                                    node_properties_to_project: List[str],
                                    edge_properties_to_project: List[str],
                                    directed = True,
                                    node_dimension = 1,
                                    edge_dimension = 1) -> None:

        if node_dimension < 1 or edge_dimension < 1:
            raise ValueError("Dimensions must be positive integers")

        try:
            nodes_data, edges_data = await adapter.get_graph_data()

            if not nodes_data:
                raise ValueError("No node data retrieved from the database.")
            if not edges_data:
                raise ValueError("No edge data retrieved from the database.")

            for node_id, properties in nodes_data:
                node_attributes = {key: properties.get(key) for key in node_properties_to_project}
                self.add_node(Node(str(node_id), node_attributes, dimension=node_dimension))

            for source_id, target_id, relationship_type, properties in edges_data:
                source_node = self.get_node(str(source_id))
                target_node = self.get_node(str(target_id))
                if source_node and target_node:
                    edge_attributes = {key: properties.get(key) for key in edge_properties_to_project}
                    edge_attributes['relationship_type'] = relationship_type

                    edge = Edge(source_node, target_node, attributes=edge_attributes, directed=directed, dimension=edge_dimension)
                    self.add_edge(edge)

                    source_node.add_skeleton_edge(edge)
                    target_node.add_skeleton_edge(edge)

                else:
                    raise ValueError(f"Edge references nonexistent nodes: {source_id} -> {target_id}")

        except (ValueError, TypeError) as e:
            print(f"Error projecting graph: {e}")
        except Exception as ex:
            print(f"Unexpected error: {ex}")
