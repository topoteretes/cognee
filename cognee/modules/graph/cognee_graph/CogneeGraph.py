import time
from cognee.shared.logging_utils import get_logger
from typing import List, Dict, Union, Optional, Type, Iterable, Tuple, Callable, Any

from cognee.modules.graph.exceptions import (
    EntityNotFoundError,
    EntityAlreadyExistsError,
    InvalidDimensionsError,
)
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge
from cognee.modules.graph.cognee_graph.CogneeAbstractGraph import CogneeAbstractGraph
import heapq

logger = get_logger("CogneeGraph")


class CogneeGraph(CogneeAbstractGraph):
    """
    Concrete implementation of the AbstractGraph class for Cognee.

    This class provides the functionality to manage nodes and edges,
    and project a graph from a database using adapters.
    """

    nodes: Dict[str, Node]
    edges: List[Edge]
    directed: bool
    triplet_distance_penalty: float

    def __init__(self, directed: bool = True):
        self.nodes = {}
        self.edges = []
        self.directed = directed
        self.triplet_distance_penalty = 3.5

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

    async def _get_nodeset_subgraph(
        self,
        adapter,
        node_type,
        node_name,
    ):
        """Retrieve subgraph based on node type and name."""
        logger.info("Retrieving graph filtered by node type and node name (NodeSet).")
        nodes_data, edges_data = await adapter.get_nodeset_subgraph(
            node_type=node_type, node_name=node_name
        )
        if not nodes_data or not edges_data:
            raise EntityNotFoundError(
                message="Nodeset does not exist, or empty nodeset projected from the database."
            )
        return nodes_data, edges_data

    async def _get_full_or_id_filtered_graph(
        self,
        adapter,
        relevant_ids_to_filter,
    ):
        """Retrieve full or ID-filtered graph with fallback."""
        if relevant_ids_to_filter is None:
            logger.info("Retrieving full graph.")
            nodes_data, edges_data = await adapter.get_graph_data()
            if not nodes_data or not edges_data:
                raise EntityNotFoundError(message="Empty graph projected from the database.")
            return nodes_data, edges_data

        get_graph_data_fn = getattr(adapter, "get_id_filtered_graph_data", adapter.get_graph_data)
        if getattr(adapter.__class__, "get_id_filtered_graph_data", None):
            logger.info("Retrieving ID-filtered graph from database.")
            nodes_data, edges_data = await get_graph_data_fn(target_ids=relevant_ids_to_filter)
        else:
            logger.info("Retrieving full graph from database.")
            nodes_data, edges_data = await get_graph_data_fn()
        if hasattr(adapter, "get_id_filtered_graph_data") and (not nodes_data or not edges_data):
            logger.warning(
                "Id filtered graph returned empty, falling back to full graph retrieval."
            )
            logger.info("Retrieving full graph")
            nodes_data, edges_data = await adapter.get_graph_data()

        if not nodes_data or not edges_data:
            raise EntityNotFoundError("Empty graph projected from the database.")
        return nodes_data, edges_data

    async def _get_filtered_graph(
        self,
        adapter,
        memory_fragment_filter,
    ):
        """Retrieve graph filtered by attributes."""
        logger.info("Retrieving graph filtered by memory fragment")
        nodes_data, edges_data = await adapter.get_filtered_graph_data(
            attribute_filters=memory_fragment_filter
        )
        if not nodes_data or not edges_data:
            raise EntityNotFoundError(message="Empty filtered graph projected from the database.")
        return nodes_data, edges_data

    async def project_graph_from_db(
        self,
        adapter: Union[GraphDBInterface],
        node_properties_to_project: List[str],
        edge_properties_to_project: List[str],
        directed=True,
        node_dimension=1,
        edge_dimension=1,
        memory_fragment_filter=[],
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
        relevant_ids_to_filter: Optional[List[str]] = None,
        triplet_distance_penalty: float = 3.5,
    ) -> None:
        if node_dimension < 1 or edge_dimension < 1:
            raise InvalidDimensionsError()
        try:
            if node_type is not None and node_name not in [None, [], ""]:
                nodes_data, edges_data = await self._get_nodeset_subgraph(
                    adapter, node_type, node_name
                )
            elif len(memory_fragment_filter) == 0:
                nodes_data, edges_data = await self._get_full_or_id_filtered_graph(
                    adapter, relevant_ids_to_filter
                )
            else:
                nodes_data, edges_data = await self._get_filtered_graph(
                    adapter, memory_fragment_filter
                )

            self.triplet_distance_penalty = triplet_distance_penalty

            import time

            start_time = time.time()
            # Process nodes
            for node_id, properties in nodes_data:
                node_attributes = {key: properties.get(key) for key in node_properties_to_project}
                self.add_node(
                    Node(
                        str(node_id),
                        node_attributes,
                        dimension=node_dimension,
                        node_penalty=triplet_distance_penalty,
                    )
                )

            # Process edges
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
                        edge_penalty=triplet_distance_penalty,
                    )
                    self.add_edge(edge)

                    source_node.add_skeleton_edge(edge)
                    target_node.add_skeleton_edge(edge)
                else:
                    raise EntityNotFoundError(
                        message=f"Edge references nonexistent nodes: {source_id} -> {target_id}"
                    )

            # Final statistics
            projection_time = time.time() - start_time
            logger.info(
                f"Graph projection completed: {len(self.nodes)} nodes, {len(self.edges)} edges in {projection_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error during graph projection: {str(e)}")
            raise

    def _initialize_vector_distance(self, graph_elements, query_list_length=None) -> None:
        """Initialize vector_distance as a list of default penalties for all graph elements."""
        query_count = query_list_length or 1
        for element in graph_elements:
            element.attributes["vector_distance"] = [self.triplet_distance_penalty] * query_count

    def _normalize_query_input(self, distance_data, query_list_length=None, name="input"):
        """Normalize single-query or multi-query input to list of lists, return empty list if empty."""
        if not distance_data:
            return []
        normalized = (
            distance_data if isinstance(distance_data[0], (list, tuple)) else [distance_data]
        )
        if query_list_length is not None and len(normalized) != query_list_length:
            raise ValueError(
                f"{name} has {len(normalized)} query lists, but query_list_length is {query_list_length}"
            )
        return normalized

    def _apply_vector_distance_updates(
        self,
        element_distances,
        query_index: int,
        get_element: Callable[[str], Optional[Union[Node, Edge]]],
        get_id_and_score: Callable[[Any], Tuple[Optional[str], Optional[float]]],
    ) -> None:
        """Apply updates into element.attributes["vector_distance"][query_index]."""
        for res in element_distances:
            key, score = get_id_and_score(res)
            if key is None or score is None:
                continue
            element = get_element(key)
            if element is None:
                continue
            element.attributes["vector_distance"][query_index] = score

    def _get_node_id_and_score(self, res: Any) -> Tuple[str, float]:
        """Extract node ID and score from a scored result."""
        return str(res.id), float(res.score)

    def _get_edge_id_and_score(self, res: Any) -> Tuple[Optional[str], Optional[float]]:
        """Extract edge key and score from a scored result."""
        payload = getattr(res, "payload", None)
        if not payload:
            return None, None
        text = payload.get("text")
        if text is None:
            return None, None
        return str(text), float(res.score)

    async def map_vector_distances_to_graph_nodes(
        self,
        node_distances,
        query_list_length: Optional[int] = None,
    ) -> None:
        self._initialize_vector_distance(self.nodes.values(), query_list_length)

        for collection_name, scored_results in node_distances.items():
            per_query_lists = self._normalize_query_input(
                scored_results, query_list_length, f"Collection '{collection_name}'"
            )
            if not per_query_lists:
                continue

            for query_index, scored_list in enumerate(per_query_lists):
                self._apply_vector_distance_updates(
                    element_distances=scored_list,
                    query_index=query_index,
                    get_element=self.nodes.get,
                    get_id_and_score=self._get_node_id_and_score,
                )

    async def map_vector_distances_to_graph_edges(
        self,
        edge_distances,
        query_list_length: Optional[int] = None,
    ) -> None:
        try:
            self._initialize_vector_distance(self.edges, query_list_length)

            normalized_edges = self._normalize_query_input(
                edge_distances, query_list_length, "edge_distances"
            )
            if not normalized_edges:
                return

            edges_by_key: Dict[str, Edge] = {}
            for edge in self.edges:
                key = edge.attributes.get("edge_text") or edge.attributes.get("relationship_type")
                if key:
                    edges_by_key[str(key)] = edge

            for query_index, scored_list in enumerate(normalized_edges):
                self._apply_vector_distance_updates(
                    element_distances=scored_list,
                    query_index=query_index,
                    get_element=edges_by_key.get,
                    get_id_and_score=self._get_edge_id_and_score,
                )

        except Exception as ex:
            logger.error(f"Error mapping vector distances to edges: {str(ex)}")
            raise ex

    def _calculate_query_top_triplet_importances(
        self,
        k: int,
        query_index: int = 0,
    ) -> List[Edge]:
        """Calculate top k triplet importances for a specific query index."""

        def score(edge):
            distances = [
                edge.node1.attributes.get("vector_distance"),
                edge.node2.attributes.get("vector_distance"),
                edge.attributes.get("vector_distance"),
            ]
            return sum(float(d[query_index]) for d in distances)

        return heapq.nsmallest(k, self.edges, key=score)

    async def calculate_top_triplet_importances(
        self, k: int, query_list_length: Optional[int] = None
    ) -> Union[List[Edge], List[List[Edge]]]:
        """Calculate top k triplet importances, supporting both single and multi-query modes."""
        query_count = query_list_length or 1
        results = [
            self._calculate_query_top_triplet_importances(k=k, query_index=i)
            for i in range(query_count)
        ]

        if query_list_length is None:
            return results[0]
        return results
