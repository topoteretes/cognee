from typing import List, Tuple

from pydantic import BaseModel, ConfigDict

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node


class Path(BaseModel):
    """
    A connected walk through the graph.

    Attributes:
        nodes (List[Node]): Ordered nodes visited by the walk (len(edges) + 1 entries).
        edges (List[Edge]): Ordered edges traversed by the walk; edge i connects
            nodes[i] and nodes[i + 1].
        score (float): Query-relevance score for the whole path (lower is better).
    """

    nodes: List[Node]
    edges: List[Edge]
    score: float = 0.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def dedup_key(self) -> Tuple[str, ...]:
        """Deterministic identity of the path: the sequence of traversed edge IDs."""
        return tuple(edge.stable_id() for edge in self.edges)

    def __len__(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        node_ids = " -> ".join(node.id for node in self.nodes)
        return f"Path({node_ids}, score={self.score})"
