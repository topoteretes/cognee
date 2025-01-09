import numpy as np

from typing import List, Dict, Union

from cognee.exceptions import InvalidValueError
from cognee.modules.graph.exceptions import EntityNotFoundError, EntityAlreadyExistsError
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge
from cognee.modules.graph.cognee_graph.CogneeAbstractGraph import CogneeAbstractGraph
import heapq
from graphistry import edges


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
        if edge not in self.edges:
            self.edges.append(edge)
            edge.node1.add_skeleton_edge(edge)
            edge.node2.add_skeleton_edge(edge)
        else:
            print(f"Edge {edge} already exists in the graph.")

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

            for node_id, properties in nodes_data:
                node_attributes = {key: properties.get(key) for key in node_properties_to_project}
                self.add_node(Node(str(node_id), node_attributes, dimension=node_dimension))

            for source_id, target_id, relationship_type, properties in edges_data:
                source_node = self.get_node(str(source_id))
                target_node = self.get_node(str(target_id))
                if source_node and target_node:
                    edge_attributes = {
                        key: properties.get(key) for key in edge_properties_to_project
                    }
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

                else:
                    raise EntityNotFoundError(
                        message=f"Edge references nonexistent nodes: {source_id} -> {target_id}"
                    )

        except (ValueError, TypeError) as e:
            print(f"Error projecting graph: {e}")
        except Exception as ex:
            print(f"Unexpected error: {ex}")

    async def map_vector_distances_to_graph_nodes(self, node_distances) -> None:
        for category, scored_results in node_distances.items():
            for scored_result in scored_results:
                node_id = str(scored_result.id)
                score = scored_result.score
                node = self.get_node(node_id)
                if node:
                    node.add_attribute("vector_distance", score)
                else:
                    print(f"Node with id {node_id} not found in the graph.")

    async def map_vector_distances_to_graph_edges(
        self, vector_engine, query
    ) -> None:  # :TODO: When we calculate edge embeddings in vector db change this similarly to node mapping
        try:
            # Step 1: Generate the query embedding
            query_vector = await vector_engine.embed_data([query])
            query_vector = query_vector[0]
            if query_vector is None or len(query_vector) == 0:
                raise ValueError("Failed to generate query embedding.")

            # Step 2: Collect all unique relationship types
            unique_relationship_types = set()
            for edge in self.edges:
                relationship_type = edge.attributes.get("relationship_type")
                if relationship_type:
                    unique_relationship_types.add(relationship_type)

            # Step 3: Embed all unique relationship types
            unique_relationship_types = list(unique_relationship_types)
            relationship_type_embeddings = await vector_engine.embed_data(unique_relationship_types)

            # Step 4: Map relationship types to their embeddings and calculate distances
            embedding_map = {}
            for relationship_type, embedding in zip(
                unique_relationship_types, relationship_type_embeddings
            ):
                edge_vector = np.array(embedding)

                # Calculate cosine similarity
                similarity = np.dot(query_vector, edge_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(edge_vector)
                )
                distance = 1 - similarity

                # Round the distance to 4 decimal places and store it
                embedding_map[relationship_type] = round(distance, 4)

            # Step 4: Assign precomputed distances to edges
            for edge in self.edges:
                relationship_type = edge.attributes.get("relationship_type")
                if not relationship_type or relationship_type not in embedding_map:
                    print(f"Edge {edge} has an unknown or missing relationship type.")
                    continue

                # Assign the precomputed distance
                edge.attributes["vector_distance"] = embedding_map[relationship_type]

        except Exception as ex:
            print(f"Error mapping vector distances to edges: {ex}")

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
