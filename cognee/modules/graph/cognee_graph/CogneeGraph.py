import time
from cognee.shared.logging_utils import get_logger
from typing import List, Dict, Union, Optional, Type

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
        if node_dimension < 1 or edge_dimension < 1:
            raise InvalidDimensionsError()
        try:
            import time

            start_time = time.time()

            # Determine projection strategy
            if node_type is not None and node_name not in [None, []]:
                nodes_data, edges_data = await adapter.get_nodeset_subgraph(
                    node_type=node_type, node_name=node_name
                )
                if not nodes_data or not edges_data:
                    raise EntityNotFoundError(
                        message="Nodeset does not exist, or empty nodetes projected from the database."
                    )
            elif len(memory_fragment_filter) == 0:
                nodes_data, edges_data = await adapter.get_graph_data()
                if not nodes_data or not edges_data:
                    raise EntityNotFoundError(message="Empty graph projected from the database.")
            else:
                nodes_data, edges_data = await adapter.get_filtered_graph_data(
                    attribute_filters=memory_fragment_filter
                )
                if not nodes_data or not edges_data:
                    raise EntityNotFoundError(
                        message="Empty filtered graph projected from the database."
                    )

            # Process nodes
            for node_id, properties in nodes_data:
                node_attributes = {key: properties.get(key) for key in node_properties_to_project}
                self.add_node(Node(str(node_id), node_attributes, dimension=node_dimension))

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

    async def map_vector_distances_to_graph_nodes(self, node_distances) -> None:
        mapped_nodes = 0
        for category, scored_results in node_distances.items():
            for scored_result in scored_results:
                node_id = str(scored_result.id)
                score = scored_result.score
                node = self.get_node(node_id)
                if node:
                    node.add_attribute("vector_distance", score)
                    mapped_nodes += 1

    async def map_vector_distances_to_graph_edges(
        self, vector_engine, query_vector, edge_distances
    ) -> None:
        try:
            if query_vector is None or len(query_vector) == 0:
                raise ValueError("Failed to generate query embedding.")

            if edge_distances is None:
                start_time = time.time()
                edge_distances = await vector_engine.search(
                    collection_name="EdgeType_relationship_name",
                    query_vector=query_vector,
                    limit=0,
                )
                projection_time = time.time() - start_time
                logger.info(
                    f"Edge collection distances were calculated separately from nodes in {projection_time:.2f}s"
                )

            embedding_map = {result.payload["text"]: result.score for result in edge_distances}

            for edge in self.edges:
                relationship_type = edge.attributes.get("relationship_type")
                distance = embedding_map.get(relationship_type, None)
                if distance is not None:
                    edge.attributes["vector_distance"] = distance

        except Exception as ex:
            logger.error(f"Error mapping vector distances to edges: {str(ex)}")
            raise ex

    async def calculate_top_triplet_importances(self, k: int) -> List:
        def score(edge):
            n1 = edge.node1.attributes.get("vector_distance", 1)
            n2 = edge.node2.attributes.get("vector_distance", 1)
            e = edge.attributes.get("vector_distance", 1)
            return n1 + n2 + e

        return heapq.nsmallest(k, self.edges, key=score)

    @staticmethod
    async def resolve_edges_to_text(retrieved_edges: list) -> str:
        """
        Converts retrieved graph edges into a human-readable string format.

        Parameters:
        -----------

            - retrieved_edges (list): A list of edges retrieved from the graph.

        Returns:
        --------

            - str: A formatted string representation of the nodes and their connections.
        """

        def _get_nodes(retrieved_edges: list) -> dict:
            def _get_title(text: str, first_n_words: int = 7, top_n_words: int = 3) -> str:
                def _top_n_words(text, stop_words=None, top_n=3, separator=", "):
                    """Concatenates the top N frequent words in text."""
                    if stop_words is None:
                        from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS

                        stop_words = DEFAULT_STOP_WORDS

                    import string

                    words = [word.lower().strip(string.punctuation) for word in text.split()]

                    if stop_words:
                        words = [word for word in words if word and word not in stop_words]

                    from collections import Counter

                    top_words = [word for word, freq in Counter(words).most_common(top_n)]

                    return separator.join(top_words)

                """Creates a title, by combining first words with most frequent words from the text."""
                first_n_words = text.split()[:first_n_words]
                top_n_words = _top_n_words(text, top_n=top_n_words)
                return f"{' '.join(first_n_words)}... [{top_n_words}]"

            """Creates a dictionary of nodes with their names and content."""
            nodes = {}
            for edge in retrieved_edges:
                for node in (edge.node1, edge.node2):
                    if node.id not in nodes:
                        text = node.attributes.get("text")
                        if text:
                            name = _get_title(text)
                            content = text
                        else:
                            name = node.attributes.get("name", "Unnamed Node")
                            content = node.attributes.get("description", name)
                        nodes[node.id] = {"node": node, "name": name, "content": content}
            return nodes

        nodes = _get_nodes(retrieved_edges)
        node_section = "\n".join(
            f"Node: {info['name']}\n__node_content_start__\n{info['content']}\n__node_content_end__\n"
            for info in nodes.values()
        )
        connection_section = "\n".join(
            f"{nodes[edge.node1.id]['name']} --[{edge.attributes['relationship_type']}]--> {nodes[edge.node2.id]['name']}"
            for edge in retrieved_edges
        )
        return f"Nodes:\n{node_section}\n\nConnections:\n{connection_section}"
