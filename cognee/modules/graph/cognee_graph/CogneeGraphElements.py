import numpy as np
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, ConfigDict, field_serializer
from cognee.modules.graph.exceptions import InvalidDimensionsError, DimensionOutOfRangeError


class Node(BaseModel):
    """
    Represents a node in a graph.
    Attributes:
        id (str): A unique identifier for the node.
        attributes (Dict[str, Any]): A dictionary of attributes associated with the node.
        neighbors (List[Node]): Represents the original nodes
        skeleton_edges (List[Edge]): Represents the original edges
    """

    id: str
    attributes: Dict[str, Any]
    skeleton_neighbours: List["Node"]
    skeleton_edges: List["Edge"]
    status: np.ndarray
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        node_id: str,
        attributes: Optional[Dict[str, Any]] = None,
        dimension: int = 1,
        node_penalty: float = 6.5,
    ):
        if dimension <= 0:
            raise InvalidDimensionsError()
        node_attributes = attributes if attributes is not None else {}
        node_attributes["vector_distance"] = None
        super().__init__(
            id=node_id,
            attributes=node_attributes,
            skeleton_neighbours=[],
            skeleton_edges=[],
            status=np.ones(dimension, dtype=int),
        )

    @field_serializer("status")
    def serialize_status(self, status: np.ndarray, _info) -> List[int]:
        return status.tolist()

    @field_serializer("skeleton_neighbours")
    def serialize_skeleton_neighbours(
        self, neighbours: List["Node"], _info
    ) -> List[Dict[str, Any]]:
        return [n.to_json() for n in neighbours]

    @field_serializer("skeleton_edges")
    def serialize_skeleton_edges(self, edges: List["Edge"], _info) -> List[Dict[str, Any]]:
        return [e.to_json() for e in edges]

    def reset_vector_distances(self, query_count: int, default_penalty: float) -> None:
        self.attributes["vector_distance"] = [default_penalty] * query_count

    def ensure_vector_distance_list(self, query_count: int, default_penalty: float) -> List[float]:
        distances = self.attributes.get("vector_distance")
        if not isinstance(distances, list) or len(distances) != query_count:
            distances = [default_penalty] * query_count
            self.attributes["vector_distance"] = distances
        return distances

    def update_distance_for_query(
        self,
        query_index: int,
        score: float,
        query_count: int,
        default_penalty: float,
    ) -> None:
        distances = self.ensure_vector_distance_list(query_count, default_penalty)
        distances[query_index] = score

    def add_skeleton_neighbor(self, neighbor: "Node") -> None:
        if neighbor not in self.skeleton_neighbours:
            self.skeleton_neighbours.append(neighbor)

    def remove_skeleton_neighbor(self, neighbor: "Node") -> None:
        if neighbor in self.skeleton_neighbours:
            self.skeleton_neighbours.remove(neighbor)

    def add_skeleton_edge(self, edge: "Edge") -> None:
        self.skeleton_edges.append(edge)
        # Add neighbor
        if edge.node1 == self:
            self.add_skeleton_neighbor(edge.node2)
        elif edge.node2 == self:
            self.add_skeleton_neighbor(edge.node1)

    def remove_skeleton_edge(self, edge: "Edge") -> None:
        if edge in self.skeleton_edges:
            self.skeleton_edges.remove(edge)
            # Remove neighbor if no other edge connects them
            neighbor = edge.node2 if edge.node1 == self else edge.node1
            if all(e.node1 != neighbor and e.node2 != neighbor for e in self.skeleton_edges):
                self.remove_skeleton_neighbor(neighbor)

    def is_node_alive_in_dimension(self, dimension: int) -> bool:
        if dimension < 0 or dimension >= len(self.status):
            raise DimensionOutOfRangeError(dimension=dimension, max_index=len(self.status) - 1)
        return self.status[dimension] == 1

    def add_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def get_attribute(self, key: str) -> Union[str, int, float]:
        return self.attributes[key]

    def get_skeleton_edges(self):
        return self.skeleton_edges

    def get_skeleton_neighbours(self):
        return self.skeleton_neighbours

    def to_json(self) -> Dict[str, Any]:
        return {
            "node_id": self.id,
            "node_attributes": self.attributes,
        }

    def __repr__(self) -> str:
        return f"Node({self.id}, attributes={self.attributes})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: "Node") -> bool:
        return isinstance(other, Node) and self.id == other.id


class Edge(BaseModel):
    """
    Represents an edge in a graph, connecting two nodes.
    Attributes:
        node1 (Node): The starting node of the edge.
        node2 (Node): The ending node of the edge.
        attributes (Dict[str, Any]): A dictionary of attributes associated with the edge.
        directed (bool): A flag indicating whether the edge is directed or undirected.
    """

    node1: "Node"
    node2: "Node"
    attributes: Dict[str, Any]
    directed: bool
    status: np.ndarray
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        node1: "Node",
        node2: "Node",
        attributes: Optional[Dict[str, Any]] = None,
        directed: bool = True,
        dimension: int = 1,
        edge_penalty: float = 6.5,
    ):
        if dimension <= 0:
            raise InvalidDimensionsError()
        edge_attributes = attributes if attributes is not None else {}
        edge_attributes["vector_distance"] = None
        super().__init__(
            node1=node1,
            node2=node2,
            attributes=edge_attributes,
            directed=directed,
            status=np.ones(dimension, dtype=int),
        )

    @field_serializer("status")
    def serialize_status(self, status: np.ndarray, _info) -> List[int]:
        return status.tolist()

    @field_serializer("node1", "node2")
    def serialize_node(self, node: "Node", _info) -> Dict[str, Any]:
        return node.to_json()

    def get_distance_key(self) -> Optional[str]:
        key = self.attributes.get("edge_type_id")
        if key is None:
            return None
        return str(key)

    def reset_vector_distances(self, query_count: int, default_penalty: float) -> None:
        self.attributes["vector_distance"] = [default_penalty] * query_count

    def ensure_vector_distance_list(self, query_count: int, default_penalty: float) -> List[float]:
        distances = self.attributes.get("vector_distance")
        if not isinstance(distances, list) or len(distances) != query_count:
            distances = [default_penalty] * query_count
            self.attributes["vector_distance"] = distances
        return distances

    def update_distance_for_query(
        self,
        query_index: int,
        score: float,
        query_count: int,
        default_penalty: float,
    ) -> None:
        distances = self.ensure_vector_distance_list(query_count, default_penalty)
        distances[query_index] = score

    def is_edge_alive_in_dimension(self, dimension: int) -> bool:
        if dimension < 0 or dimension >= len(self.status):
            raise DimensionOutOfRangeError(dimension=dimension, max_index=len(self.status) - 1)
        return self.status[dimension] == 1

    def add_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def get_attribute(self, key: str) -> Optional[Union[str, int, float]]:
        return self.attributes.get(key)

    def get_source_node(self):
        return self.node1

    def get_destination_node(self):
        return self.node2

    def to_json(self) -> Dict[str, Any]:
        return {
            "source_node_id": self.node1.id,
            "target_node_id": self.node2.id,
            "source_node_attributes": self.node1.attributes,
            "target_node_attributes": self.node2.attributes,
            "edge_attributes": self.attributes,
        }

    def __repr__(self) -> str:
        direction = "->" if self.directed else "--"
        return f"Edge({self.node1.id} {direction} {self.node2.id}, attributes={self.attributes})"

    def __hash__(self) -> int:
        if self.directed:
            return hash((self.node1, self.node2))
        else:
            return hash(frozenset({self.node1, self.node2}))

    def __eq__(self, other: "Edge") -> bool:
        if not isinstance(other, Edge):
            return False
        if self.directed:
            return self.node1 == other.node1 and self.node2 == other.node2
        else:
            return {self.node1, self.node2} == {other.node1, other.node2}
