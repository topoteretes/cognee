from abc import ABC, abstractmethod

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node


class CogneeAbstractGraph(ABC):
    """
    Abstract base class for representing a graph structure.

    """

    @abstractmethod
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""

    @abstractmethod
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""

    @abstractmethod
    def get_node(self, node_id: str) -> Node:
        """Retrieve a node by its ID."""

    @abstractmethod
    def get_edges(self, node_id: str) -> list[Edge]:
        """Retrieve edges connected to a specific node."""

    @abstractmethod
    async def project_graph_from_db(
        self, adapter: GraphDBInterface, directed: bool, dimension: int
    ) -> None:
        """Project the graph structure from a database using the provided adapter."""
