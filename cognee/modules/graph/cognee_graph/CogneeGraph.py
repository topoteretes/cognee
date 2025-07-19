from cognee.shared.logging_utils import get_logger
from typing import List, Dict, Union, Optional, Type

from cognee.exceptions import InvalidValueError
from cognee.modules.graph.exceptions import EntityNotFoundError, EntityAlreadyExistsError
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

    def __init__(self, directed: bool = True):
        self.nodes = {}
        self.edges = []
        self.directed = directed
        logger.debug(f"Initialized CogneeGraph with directed={directed}")

    def add_node(self, node: Node) -> None:
        if node.id not in self.nodes:
            self.nodes[node.id] = node
            logger.debug(f"Added node {node.id} to graph")
        else:
            logger.warning(f"Attempted to add duplicate node {node.id}")
            raise EntityAlreadyExistsError(message=f"Node with id {node.id} already exists.")

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        edge.node1.add_skeleton_edge(edge)
        edge.node2.add_skeleton_edge(edge)
        logger.debug(
            f"Added edge {edge.node1.id} -> {edge.node2.id} with relationship: {edge.attributes.get('relationship_type', 'UNKNOWN')}"
        )

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
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
    ) -> None:
        logger.info("Starting graph projection from database")
        logger.debug(
            f"Projection parameters: directed={directed}, node_dimension={node_dimension}, edge_dimension={edge_dimension}"
        )
        logger.debug(f"Node properties to project: {node_properties_to_project}")
        logger.debug(f"Edge properties to project: {edge_properties_to_project}")

        if node_dimension < 1 or edge_dimension < 1:
            logger.error(
                f"Invalid dimensions: node_dimension={node_dimension}, edge_dimension={edge_dimension}"
            )
            raise InvalidValueError(message="Dimensions must be positive integers")

        try:
            import time

            start_time = time.time()

            # Determine projection strategy and log it
            if node_type is not None and node_name is not None:
                logger.info(
                    f"Using nodeset subgraph projection strategy for node_type={node_type.__name__}, node_names={node_name}"
                )
                nodes_data, edges_data = await adapter.get_nodeset_subgraph(
                    node_type=node_type, node_name=node_name
                )
                if not nodes_data or not edges_data:
                    logger.warning("Nodeset subgraph projection returned empty results")
                    raise EntityNotFoundError(
                        message="Nodeset does not exist, or empty nodetes projected from the database."
                    )
                logger.info(
                    f"Nodeset subgraph projection retrieved {len(nodes_data)} nodes and {len(edges_data)} edges"
                )

            elif len(memory_fragment_filter) == 0:
                logger.info("Using full graph projection strategy")
                nodes_data, edges_data = await adapter.get_graph_data()
                if not nodes_data or not edges_data:
                    logger.warning("Full graph projection returned empty results")
                    raise EntityNotFoundError(message="Empty graph projected from the database.")
                logger.info(
                    f"Full graph projection retrieved {len(nodes_data)} nodes and {len(edges_data)} edges"
                )

            else:
                logger.info(
                    f"Using filtered graph projection strategy with {len(memory_fragment_filter)} filter(s)"
                )
                logger.debug(f"Memory fragment filters: {memory_fragment_filter}")
                nodes_data, edges_data = await adapter.get_filtered_graph_data(
                    attribute_filters=memory_fragment_filter
                )

                if not nodes_data or not edges_data:
                    logger.warning("Filtered graph projection returned empty results")
                    raise EntityNotFoundError(
                        message="Empty filtered graph projected from the database."
                    )
                logger.info(
                    f"Filtered graph projection retrieved {len(nodes_data)} nodes and {len(edges_data)} edges"
                )

            # Process nodes
            logger.debug("Starting node processing phase")
            processed_nodes = 0
            failed_nodes = 0

            for node_id, properties in nodes_data:
                try:
                    node_attributes = {
                        key: properties.get(key) for key in node_properties_to_project
                    }
                    self.add_node(Node(str(node_id), node_attributes, dimension=node_dimension))
                    processed_nodes += 1
                except Exception as e:
                    failed_nodes += 1
                    logger.warning(f"Failed to process node {node_id}: {str(e)}")

            logger.info(
                f"Node processing completed: {processed_nodes} processed, {failed_nodes} failed"
            )

            # Process edges
            logger.debug("Starting edge processing phase")
            processed_edges = 0
            failed_edges = 0
            missing_node_errors = 0

            for source_id, target_id, relationship_type, properties in edges_data:
                try:
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
                        processed_edges += 1
                    else:
                        missing_node_errors += 1
                        logger.warning(
                            f"Edge references nonexistent nodes: {source_id} -> {target_id}"
                        )
                        if not source_node:
                            logger.debug(f"Source node {source_id} not found")
                        if not target_node:
                            logger.debug(f"Target node {target_id} not found")

                        raise EntityNotFoundError(
                            message=f"Edge references nonexistent nodes: {source_id} -> {target_id}"
                        )
                except Exception as e:
                    failed_edges += 1
                    logger.warning(f"Failed to process edge {source_id} -> {target_id}: {str(e)}")

            logger.info(
                f"Edge processing completed: {processed_edges} processed, {failed_edges} failed, {missing_node_errors} missing node errors"
            )

            # Final statistics
            projection_time = time.time() - start_time
            logger.info(f"Graph projection completed successfully in {projection_time:.2f} seconds")
            logger.info(f"Final graph stats: {len(self.nodes)} nodes, {len(self.edges)} edges")
            logger.debug(
                f"Graph density: {len(self.edges) / (len(self.nodes) * (len(self.nodes) - 1)) if len(self.nodes) > 1 else 0:.4f}"
            )

        except (ValueError, TypeError) as e:
            logger.error(f"Error projecting graph: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during graph projection: {str(e)}")
            raise

    async def map_vector_distances_to_graph_nodes(self, node_distances) -> None:
        logger.info("Starting vector distance mapping to graph nodes")
        mapped_nodes = 0

        for category, scored_results in node_distances.items():
            logger.debug(
                f"Processing category '{category}' with {len(scored_results)} scored results"
            )
            for scored_result in scored_results:
                node_id = str(scored_result.id)
                score = scored_result.score
                node = self.get_node(node_id)
                if node:
                    node.add_attribute("vector_distance", score)
                    mapped_nodes += 1
                    logger.debug(f"Mapped vector distance {score} to node {node_id}")
                else:
                    logger.warning(f"Node {node_id} not found in graph for vector distance mapping")

        logger.info(f"Vector distance mapping completed: {mapped_nodes} nodes updated")

    async def map_vector_distances_to_graph_edges(self, vector_engine, query) -> None:
        logger.info(
            f"Starting vector distance mapping to graph edges for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        try:
            query_vector = await vector_engine.embed_data([query])
            query_vector = query_vector[0]
            if query_vector is None or len(query_vector) == 0:
                logger.error("Failed to generate query embedding for edge distance mapping")
                raise ValueError("Failed to generate query embedding.")

            logger.debug("Successfully generated query embedding for edge distance mapping")

            edge_distances = await vector_engine.search(
                collection_name="EdgeType_relationship_name",
                query_text=query,
                limit=0,
            )

            logger.debug(
                f"Retrieved {len(edge_distances)} edge distance results from vector search"
            )

            embedding_map = {result.payload["text"]: result.score for result in edge_distances}
            logger.debug(f"Created embedding map with {len(embedding_map)} entries")

            mapped_edges = 0
            missing_relationships = 0

            for edge in self.edges:
                relationship_type = edge.attributes.get("relationship_type")
                if not relationship_type or relationship_type not in embedding_map:
                    missing_relationships += 1
                    logger.debug(
                        f"Edge {edge.node1.id} -> {edge.node2.id} has unknown or missing relationship type: {relationship_type}"
                    )
                    continue

                edge.attributes["vector_distance"] = embedding_map[relationship_type]
                mapped_edges += 1
                logger.debug(
                    f"Mapped vector distance {embedding_map[relationship_type]} to edge {edge.node1.id} -> {edge.node2.id}"
                )

            logger.info(
                f"Edge vector distance mapping completed: {mapped_edges} edges updated, {missing_relationships} missing relationships"
            )

        except Exception as ex:
            logger.error(f"Error mapping vector distances to edges: {str(ex)}")
            raise ex

    async def calculate_top_triplet_importances(self, k: int) -> List:
        logger.info(f"Calculating top {k} triplet importances")
        logger.debug(f"Total edges to evaluate: {len(self.edges)}")

        min_heap = []
        processed_triplets = 0

        for i, edge in enumerate(self.edges):
            source_node = self.get_node(edge.node1.id)
            target_node = self.get_node(edge.node2.id)

            source_distance = source_node.attributes.get("vector_distance", 1) if source_node else 1
            target_distance = target_node.attributes.get("vector_distance", 1) if target_node else 1
            edge_distance = edge.attributes.get("vector_distance", 1)

            total_distance = source_distance + target_distance + edge_distance
            processed_triplets += 1

            heapq.heappush(min_heap, (-total_distance, i, edge))
            if len(min_heap) > k:
                heapq.heappop(min_heap)

        result = [edge for _, _, edge in sorted(min_heap)]
        logger.info(
            f"Top triplet calculation completed: processed {processed_triplets} triplets, returning top {len(result)}"
        )
        logger.debug(f"Top triplet scores: {[-score for score, _, _ in sorted(min_heap)]}")

        return result
