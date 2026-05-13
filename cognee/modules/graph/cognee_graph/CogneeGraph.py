import time
from cognee.shared.logging_utils import get_logger
from cognee.modules.engine.utils.generate_edge_id import generate_edge_id
from typing import List, Dict, Union, Optional, Type, Iterable, Tuple, Callable, Any

from cognee.modules.graph.exceptions import (
    EntityNotFoundError,
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
    edges_by_distance_key: Dict[str, List[Edge]]
    directed: bool
    triplet_distance_penalty: float
    feedback_influence: float

    def __init__(self, directed: bool = True):
        self.nodes = {}
        self.edges = []
        self.edges_by_distance_key = {}
        self.directed = directed
        self.triplet_distance_penalty = 6.5
        self.feedback_influence = 0.0

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            logger.debug(
                "Skipping duplicate node",
                extra={"node_id": node.id},
            )
            return
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

        edge_text = edge.attributes.get("edge_text") or edge.attributes.get("relationship_type")
        edge.attributes["edge_type_id"] = (
            generate_edge_id(edge_id=edge_text) if edge_text else None
        )  # Update edge with generated edge_type_id

        edge.node1.add_skeleton_edge(edge)
        edge.node2.add_skeleton_edge(edge)
        key = edge.get_distance_key()
        if not key:
            return
        if key not in self.edges_by_distance_key:
            self.edges_by_distance_key[key] = []
        self.edges_by_distance_key[key].append(edge)

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

    def reset_distances(self, collection: Iterable[Union[Node, Edge]], query_count: int) -> None:
        """Reset vector distances for a collection of nodes or edges."""
        for item in collection:
            item.reset_vector_distances(query_count, self.triplet_distance_penalty)

    def _normalize_query_distance_lists(
        self, distances: List, query_list_length: Optional[int] = None, name: str = "distances"
    ) -> List:
        """Normalize shape: flat list -> single-query; nested list -> multi-query."""
        if not distances:
            return []
        first_item = distances[0]
        if isinstance(first_item, (list, tuple)):
            per_query_lists = distances
        else:
            per_query_lists = [distances]
        if query_list_length is not None and len(per_query_lists) != query_list_length:
            raise ValueError(
                f"{name} has {len(per_query_lists)} query lists, "
                f"but query_list_length is {query_list_length}"
            )
        return per_query_lists

    async def _get_nodeset_subgraph(self, adapter, node_type, node_name, node_name_filter_operator):
        """Retrieve subgraph based on node type and name."""
        logger.info("Retrieving graph filtered by node type and node name (NodeSet).")
        nodes_data, edges_data = await adapter.get_nodeset_subgraph(
            node_type=node_type,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
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

    def _process_nodes_and_edges(
        self,
        nodes_data,
        edges_data,
        node_properties_to_project: List[str],
        edge_properties_to_project: List[str],
        directed: bool,
        node_dimension: int,
        edge_dimension: int,
        triplet_distance_penalty: float,
    ) -> None:
        """Process raw node and edge data into graph elements."""
        self.triplet_distance_penalty = triplet_distance_penalty

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
                edge_attributes = {key: properties.get(key) for key in edge_properties_to_project}
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
            else:
                raise EntityNotFoundError(
                    message=f"Edge references nonexistent nodes: {source_id} -> {target_id}"
                )

        # Final statistics
        projection_time = time.time() - start_time
        logger.info(
            f"Graph projection completed: {len(self.nodes)} nodes, {len(self.edges)} edges in {projection_time:.2f}s"
        )

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
        node_name_filter_operator: str = "OR",
        relevant_ids_to_filter: Optional[List[str]] = None,
        triplet_distance_penalty: float = 6.5,
        feedback_influence: float = 0.0,
    ) -> None:
        if node_dimension < 1 or edge_dimension < 1:
            raise InvalidDimensionsError()
        try:
            if node_type is not None and node_name not in [None, [], ""]:
                nodes_data, edges_data = await self._get_nodeset_subgraph(
                    adapter, node_type, node_name, node_name_filter_operator
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
            self.feedback_influence = feedback_influence

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
                else:
                    raise EntityNotFoundError(
                        message=f"Edge references nonexistent nodes: {source_id} -> {target_id}"
                    )

            # Final statistics
            projection_time = time.time() - start_time
            logger.info(
                f"Graph projection completed: {len(self.nodes)} nodes, {len(self.edges)} edges in {projection_time:.2f}s"
            )

        except Exception:
            logger.error("Error during graph projection", exc_info=True)
            raise

    async def project_neighborhood_from_db(
        self,
        adapter: Union[GraphDBInterface],
        node_properties_to_project: List[str],
        edge_properties_to_project: List[str],
        seed_node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
        directed: bool = True,
        node_dimension: int = 1,
        edge_dimension: int = 1,
        triplet_distance_penalty: float = 6.5,
        feedback_influence: float = 0.0,
    ) -> None:
        """
        Project a neighborhood subgraph from the database around seed nodes.

        Calls adapter.get_neighborhood() and processes nodes/edges the same way
        as project_graph_from_db.
        """
        if node_dimension < 1 or edge_dimension < 1:
            raise InvalidDimensionsError()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if not seed_node_ids:
            raise ValueError("seed_node_ids must not be empty")
        try:
            logger.info(f"Retrieving {depth}-hop neighborhood for {len(seed_node_ids)} seed nodes.")
            nodes_data, edges_data = await adapter.get_neighborhood(
                node_ids=seed_node_ids,
                depth=depth,
                edge_types=edge_types,
            )

            if not nodes_data:
                raise EntityNotFoundError(message="Empty neighborhood projected from the database.")
            edges_data = edges_data or []

            self.triplet_distance_penalty = triplet_distance_penalty
            self.feedback_influence = feedback_influence

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
                else:
                    raise EntityNotFoundError(
                        message=f"Edge references nonexistent nodes: {source_id} -> {target_id}"
                    )

            projection_time = time.time() - start_time
            logger.info(
                f"Graph projection completed: {len(self.nodes)} nodes, {len(self.edges)} edges in {projection_time:.2f}s"
            )

        except Exception:
            logger.error("Error during neighborhood projection", exc_info=True)
            raise

    async def map_vector_distances_to_graph_nodes(
        self,
        node_distances,
        query_list_length: Optional[int] = None,
    ) -> None:
        """Map vector distances to nodes, supporting single- and multi-query input shapes."""

        query_count = query_list_length or 1

        self.reset_distances(self.nodes.values(), query_count)

        for collection_name, scored_results in node_distances.items():
            if not scored_results:
                continue

            per_query_scored_results = self._normalize_query_distance_lists(
                scored_results, query_list_length, f"Collection '{collection_name}'"
            )

            for query_index, scored_results in enumerate(per_query_scored_results):
                for result in scored_results:
                    node_id = str(getattr(result, "id", None))
                    if not node_id:
                        continue
                    node = self.get_node(node_id)
                    if node is None:
                        continue
                    score = float(getattr(result, "score", self.triplet_distance_penalty))
                    node.update_distance_for_query(
                        query_index=query_index,
                        score=score,
                        query_count=query_count,
                        default_penalty=self.triplet_distance_penalty,
                    )

    async def map_vector_distances_to_graph_edges(
        self,
        edge_distances,
        query_list_length: Optional[int] = None,
    ) -> None:
        """Map vector distances to graph edges, supporting single- and multi-query input shapes."""
        query_count = query_list_length or 1

        self.reset_distances(self.edges, query_count)

        if not edge_distances:
            return None

        per_query_scored_results = self._normalize_query_distance_lists(
            edge_distances, query_list_length, "edge_distances"
        )

        for query_index, scored_results in enumerate(per_query_scored_results):
            for result in scored_results:
                matching_edges = self.edges_by_distance_key.get(str(result.id))
                if not matching_edges:
                    continue
                for edge in matching_edges:
                    edge.update_distance_for_query(
                        query_index=query_index,
                        score=float(getattr(result, "score", self.triplet_distance_penalty)),
                        query_count=query_count,
                        default_penalty=self.triplet_distance_penalty,
                    )

    def _calculate_query_top_triplet_importances(
        self,
        k: int,
        query_index: int = 0,
        feedback_influence: Optional[float] = None,
    ) -> List[Edge]:
        """Calculate top k triplet importances for a specific query index."""
        active_feedback_influence = (
            self.feedback_influence if feedback_influence is None else feedback_influence
        )

        def _effective_distance(distance: float, feedback_weight: Any) -> float:
            if active_feedback_influence <= 0.0:
                return distance

            # Only blend real cosine distances in [0, 2].
            # Fallback penalties and out-of-range values must remain unchanged so
            # missing components stay ranked below valid matches.
            if distance >= self.triplet_distance_penalty or distance < 0.0 or distance > 2.0:
                return distance

            try:
                normalized_feedback_weight = float(feedback_weight)
            except (TypeError, ValueError):
                normalized_feedback_weight = 0.5

            normalized_feedback_weight = max(0.0, min(1.0, normalized_feedback_weight))
            # Blend in a normalized space (cosine distance in [0, 2] -> [0, 1]),
            # then project back to distance scale so score magnitudes stay consistent.
            normalized_distance = distance / 2.0
            blended_normalized = (1.0 - active_feedback_influence) * normalized_distance + (
                active_feedback_influence * (1.0 - normalized_feedback_weight)
            )
            return blended_normalized * 2.0

        def score(edge: Edge) -> float:
            elements = (
                (edge.node1, f"node {edge.node1.id}"),
                (edge.node2, f"node {edge.node2.id}"),
                (edge, f"edge {edge.node1.id}->{edge.node2.id}"),
            )

            importances = []
            for element, label in elements:
                distances = element.attributes.get("vector_distance")
                importance_weight = element.attributes.get("importance_weight")
                try:
                    importance_weight = float(importance_weight)
                except (TypeError, ValueError):
                    importance_weight = 0.5
                if not isinstance(distances, list) or query_index >= len(distances):
                    raise ValueError(
                        f"{label}: vector_distance must be a list with length > {query_index} "
                        f"before scoring (got {type(distances).__name__} with length "
                        f"{len(distances) if isinstance(distances, list) else 'n/a'})"
                    )
                value = distances[query_index]
                try:
                    distance = float(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{label}: vector_distance[{query_index}] must be float-like, "
                        f"got {type(value).__name__}"
                    )
                distance = (2 - importance_weight) * distance
                feedback_weight = element.attributes.get("feedback_weight", 0.5)
                importances.append(_effective_distance(distance, feedback_weight))

            return sum(importances)

        return heapq.nsmallest(k, self.edges, key=score)

    async def calculate_top_triplet_importances(
        self,
        k: int,
        query_list_length: Optional[int] = None,
        feedback_influence: Optional[float] = None,
    ) -> Union[List[Edge], List[List[Edge]]]:
        """Calculate top k triplet importances, supporting both single and multi-query modes."""
        query_count = query_list_length or 1
        results = [
            self._calculate_query_top_triplet_importances(
                k=k, query_index=i, feedback_influence=feedback_influence
            )
            for i in range(query_count)
        ]

        if query_list_length is None:
            return results[0]
        return results
