from abc import ABC, abstractmethod
from typing import List, Dict, Union
from CogneeGraphElements import Node, Edge
from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
from cognee.infrastructure.databases.graph.networkx.adapter import NetworkXAdapter

class CogneeAbstractGraph(ABC):
    """
    Abstract base class for representing a graph structure.

    Attributes:
        nodes (Dict[str, Node]): A dictionary of nodes in the graph, keyed by their ID.
        edges (List[Edge]): A list of edges in the graph.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = [] # :TODO do we need it in hashtable? Do we want to index?

    @abstractmethod
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        pass

    # :TODO Add dimension
    @abstractmethod
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Node:
        """Retrieve a node by its ID."""
        pass

    @abstractmethod
    def get_edges(self, node_id: str) -> List[Edge]:
        """Retrieve edges connected to a specific node."""
        pass

    @abstractmethod
    async def project_graph_from_db(self, adapter: Union[Neo4jAdapter, NetworkXAdapter]) -> None:
        """Project the graph structure from a database using the provided adapter."""
        pass
