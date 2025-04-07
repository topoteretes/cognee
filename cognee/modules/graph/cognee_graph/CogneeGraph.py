from cognee.shared.logging_utils import get_logger
from typing import List, Dict, Union

from cognee.exceptions import InvalidValueError
from cognee.modules.graph.exceptions import EntityNotFoundError, EntityAlreadyExistsError
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge
from cognee.modules.graph.cognee_graph.CogneeAbstractGraph import CogneeAbstractGraph
import heapq

logger = get_logger()


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
            raise EntityAlreadyExistsError(message=f"Node with id {node.id} already exists.")

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        edge.node1.add_skeleton_edge(edge)
        edge.node2.add_skeleton_edge(edge)

    def get_node(self, node_id: str) -> Node:
        return self.nodes.get(node_id, None)

    def get_edges_from_node(self, node_id: str) -> List[Edge]:
        node = self.get_node(node_id)
        if node:
            return node.skeleton_edges
        else:
            raise EntityNotFoundError(message=f"Node with id {node_id} does not exist.")

    def get_edges(self) -> List[Edge]:
        return self.edges

    async def _retrieve_graph_data(self, adapter, memory_fragment_filter):
        """Retrieve graph data from the adapter."""
        if len(memory_fragment_filter) == 0:
            nodes_data, edges_data = await adapter.get_graph_data()
        else:
            nodes_data, edges_data = await adapter.get_filtered_graph_data(
                attribute_filters=memory_fragment_filter
            )

        if not nodes_data:
            raise EntityNotFoundError(message="No node data retrieved from the database.")
        if not edges_data:
            raise EntityNotFoundError(message="No edge data retrieved from the database.")

        return nodes_data, edges_data

    def _create_nodes_from_data(self, nodes_data, node_properties_to_project, node_dimension):
        """Create node objects from database data."""
        for node_id, properties in nodes_data:
            node_attributes = {key: properties.get(key) for key in node_properties_to_project}
            self.add_node(Node(str(node_id), node_attributes, dimension=node_dimension))

    def _create_edges_from_data(
        self, edges_data, edge_properties_to_project, directed, edge_dimension
    ):
        """Create edge objects from database data."""
        for source_id, target_id, relationship_type, properties in edges_data:
            source_node = self.get_node(str(source_id))
            target_node = self.get_node(str(target_id))

            if not source_node or not target_node:
                raise EntityNotFoundError(
                    message=f"Edge references nonexistent nodes: {source_id} -> {target_id}"
                )

            edge_attributes = {key: properties.get(key) for key in edge_properties_to_project}
            edge_attributes["relationship_type"] = relationship_type

            edge = Edge(
                source_node,
                target_node,
                attributes=edge_attributes,
                directed=directed,
                dimension=edge_dimension,
            )
            self.add_edge(edge)

            source_node.add_skeleton_edge(edge)
            target_node.add_skeleton_edge(edge)

    async def project_graph_from_db(
        self,
        adapter: Union[GraphDBInterface],
        node_properties_to_project: List[str],
        edge_properties_to_project: List[str],
        directed=True,
        node_dimension=1,
        edge_dimension=1,
        memory_fragment_filter=[],
    ) -> None:
        if node_dimension < 1 or edge_dimension < 1:
            raise InvalidValueError(message="Dimensions must be positive integers")

        try:
            # Retrieve graph data
            nodes_data, edges_data = await self._retrieve_graph_data(
                adapter, memory_fragment_filter
            )

            # Create nodes
            self._create_nodes_from_data(nodes_data, node_properties_to_project, node_dimension)

            # Create edges
            self._create_edges_from_data(
                edges_data, edge_properties_to_project, directed, edge_dimension
            )

        except EntityNotFoundError as e:
            logger.error(f"Entity not found: {e}")
            raise e
        except (ValueError, TypeError) as e:
            logger.error(f"Error projecting graph: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in graph projection: {e}")
            raise e

    async def map_vector_distances_to_graph_nodes(self, node_distances) -> None:
        for category, scored_results in node_distances.items():
            for scored_result in scored_results:
                node_id = str(scored_result.id)
                score = scored_result.score
                node = self.get_node(node_id)
                if node:
                    node.add_attribute("vector_distance", score)

    async def map_vector_distances_to_graph_edges(self, vector_engine, query) -> None:
        try:
            query_vector = await vector_engine.embed_data([query])
            query_vector = query_vector[0]
            if query_vector is None or len(query_vector) == 0:
                raise ValueError("Failed to generate query embedding.")

            edge_distances = await vector_engine.get_distance_from_collection_elements(
                "EdgeType_relationship_name", query_text=query
            )

            embedding_map = {result.payload["text"]: result.score for result in edge_distances}

            for edge in self.edges:
                relationship_type = edge.attributes.get("relationship_type")
                if not relationship_type or relationship_type not in embedding_map:
                    print(f"Edge {edge} has an unknown or missing relationship type.")
                    continue

                edge.attributes["vector_distance"] = embedding_map[relationship_type]

        except Exception as ex:
            print(f"Error mapping vector distances to edges: {ex}")
            raise ex

    async def calculate_top_triplet_importances(self, k: int) -> List:
        min_heap = []
        for i, edge in enumerate(self.edges):
            source_node = self.get_node(edge.node1.id)
            target_node = self.get_node(edge.node2.id)

            source_distance = source_node.attributes.get("vector_distance", 1) if source_node else 1
            target_distance = target_node.attributes.get("vector_distance", 1) if target_node else 1
            edge_distance = edge.attributes.get("vector_distance", 1)

            total_distance = source_distance + target_distance + edge_distance

            heapq.heappush(min_heap, (-total_distance, i, edge))
            if len(min_heap) > k:
                heapq.heappop(min_heap)

        return [edge for _, _, edge in sorted(min_heap)]
