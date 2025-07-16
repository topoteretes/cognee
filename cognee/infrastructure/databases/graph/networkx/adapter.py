"""Adapter for NetworkX graph database."""

import os
import json
import asyncio
import numpy as np
from uuid import UUID
import networkx as nx
from datetime import datetime, timezone
from typing import Dict, Any, List, Union, Type, Tuple

from cognee.infrastructure.databases.exceptions.exceptions import NodesetFilterNotSupportedError
from cognee.infrastructure.files.storage import get_file_storage
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
)
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.modules.storage.utils import JSONEncoder

logger = get_logger()


class NetworkXAdapter(GraphDBInterface):
    """
    Manage a singleton instance of a graph database interface, utilizing the NetworkX
    library. Handles graph data access and manipulation, including nodes and edges
    management, persistence, and auxiliary functionalities.
    """

    _instance = None
    graph = None  # Class variable to store the singleton instance

    def __new__(cls, filename):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.filename = filename
        return cls._instance

    def __init__(self, filename="cognee_graph.pkl"):
        self.filename = filename

    async def get_graph_data(self):
        """
        Retrieve graph data including nodes and edges.

        Returns:
        --------

            A tuple containing a list of node data and a list of edge data.
        """
        await self.load_graph_from_file()
        return (list(self.graph.nodes(data=True)), list(self.graph.edges(data=True, keys=True)))

    async def query(self, query: str, params: dict):
        """
        Execute a query against the graph data. The specifics of the query execution need to be
        implemented.

        Parameters:
        -----------

            - query (str): The query string to run against the graph.
            - params (dict): Parameters for the query, if necessary.
        """
        pass

    async def has_node(self, node_id: UUID) -> bool:
        """
        Determine if a specific node exists in the graph.

        Parameters:
        -----------

            - node_id (UUID): The identifier of the node to check.

        Returns:
        --------

            - bool: True if the node exists, otherwise False.
        """
        return self.graph.has_node(node_id)

    async def add_node(self, node: DataPoint) -> None:
        """
        Add a node to the graph and persist the graph state to the file.

        Parameters:
        -----------

            - node (DataPoint): The node to be added, represented as a DataPoint object.
        """
        self.graph.add_node(node.id, **node.model_dump())

        await self.save_graph_to_file(self.filename)

    @record_graph_changes
    async def add_nodes(self, nodes: list[DataPoint]) -> None:
        """
        Bulk add multiple nodes to the graph and persist the graph state to the file.

        Parameters:
        -----------

            - nodes (list[DataPoint]): A list of DataPoint objects defining the nodes to be
              added.
        """
        nodes = [(node.id, node.model_dump()) for node in nodes]
        self.graph.add_nodes_from(nodes)
        await self.save_graph_to_file(self.filename)

    async def get_graph(self):
        """
        Retrieve the current state of the graph.

        Returns:
        --------

            The current graph instance.
        """
        return self.graph

    async def has_edge(self, from_node: str, to_node: str, edge_label: str) -> bool:
        """
        Check for the existence of a specific edge in the graph.

        Parameters:
        -----------

            - from_node (str): The identifier of the source node.
            - to_node (str): The identifier of the target node.
            - edge_label (str): The label of the edge to check.

        Returns:
        --------

            - bool: True if the edge exists, otherwise False.
        """
        return self.graph.has_edge(from_node, to_node, key=edge_label)

    async def has_edges(self, edges):
        """
        Check for the existence of multiple edges in the graph.

        Parameters:
        -----------

            - edges: A list of edges to check, defined as tuples of (from_node, to_node,
              edge_label).

        Returns:
        --------

            A list of edges that exist in the graph.
        """
        result = []

        for from_node, to_node, edge_label in edges:
            if self.graph.has_edge(from_node, to_node, edge_label):
                result.append((from_node, to_node, edge_label))

        return result

    @record_graph_changes
    async def add_edge(
        self,
        from_node: str,
        to_node: str,
        relationship_name: str,
        edge_properties: Dict[str, Any] = {},
    ) -> None:
        """
        Add a single edge to the graph and persist the graph state to the file.

        Parameters:
        -----------

            - from_node (str): The identifier of the source node for the edge.
            - to_node (str): The identifier of the target node for the edge.
            - relationship_name (str): The label for the relationship as the edge is created.
            - edge_properties (Dict[str, Any]): Additional properties for the edge, if any.
              (default {})
        """
        edge_properties["updated_at"] = datetime.now(timezone.utc)
        self.graph.add_edge(
            from_node,
            to_node,
            key=relationship_name,
            **(edge_properties if edge_properties else {}),
        )

        await self.save_graph_to_file(self.filename)

    @record_graph_changes
    async def add_edges(self, edges: list[tuple[str, str, str, dict]]) -> None:
        """
        Bulk add multiple edges to the graph and persist the graph state to the file.

        Parameters:
        -----------

            - edges (list[tuple[str, str, str, dict]]): A list of edges defined as tuples
              containing (from_node, to_node, relationship_name, edge_properties).
        """
        if not edges:
            logger.debug("No edges to add")
            return

        try:
            # Validate edge format and convert UUIDs to strings
            processed_edges = []
            for edge in edges:
                if len(edge) < 3 or len(edge) > 4:
                    raise ValueError(
                        f"Invalid edge format: {edge}. Expected (from_node, to_node, relationship_name[, properties])"
                    )

                # Convert UUIDs to strings if needed
                from_node = str(edge[0]) if isinstance(edge[0], UUID) else edge[0]
                to_node = str(edge[1]) if isinstance(edge[1], UUID) else edge[1]
                relationship_name = edge[2]

                if not all(isinstance(x, str) for x in [from_node, to_node, relationship_name]):
                    raise ValueError(
                        f"First three elements of edge must be strings or UUIDs: {edge}"
                    )

                # Process edge with updated_at timestamp
                processed_edge = (
                    from_node,
                    to_node,
                    relationship_name,
                    {
                        **(edge[3] if len(edge) == 4 else {}),
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
                processed_edges.append(processed_edge)

            # Add edges to graph
            self.graph.add_edges_from(processed_edges)
            logger.debug(f"Added {len(processed_edges)} edges to graph")

            # Save changes
            await self.save_graph_to_file(self.filename)
        except Exception as e:
            logger.error(f"Failed to add edges: {e}")
            raise

    async def get_edges(self, node_id: UUID):
        """
        Retrieve edges connected to a specific node.

        Parameters:
        -----------

            - node_id (UUID): The identifier of the node whose edges are to be retrieved.

        Returns:
        --------

            A list of edges connected to the specified node.
        """
        return list(self.graph.in_edges(node_id, data=True)) + list(
            self.graph.out_edges(node_id, data=True)
        )

    async def delete_node(self, node_id: UUID) -> None:
        """
        Remove a node and its associated edges from the graph, then persist the changes.

        Parameters:
        -----------

            - node_id (UUID): The identifier of the node to delete.
        """

        if self.graph.has_node(node_id):
            # First remove all edges connected to the node
            for edge in list(self.graph.edges(node_id, data=True)):
                source, target, data = edge
                self.graph.remove_edge(source, target, key=data.get("relationship_name"))

            # Then remove the node itself
            self.graph.remove_node(node_id)

            # Save the updated graph state
            await self.save_graph_to_file(self.filename)
        else:
            logger.error(f"Node {node_id} not found in graph")

    async def delete_nodes(self, node_ids: List[UUID]) -> None:
        """
        Bulk delete nodes from the graph and persist the changes.

        Parameters:
        -----------

            - node_ids (List[UUID]): A list of node identifiers to delete.
        """
        self.graph.remove_nodes_from(node_ids)
        await self.save_graph_to_file(self.filename)

    async def get_disconnected_nodes(self) -> List[str]:
        """
        Identify nodes that are not connected to any other nodes in the graph.

        Returns:
        --------

            - List[str]: A list of identifiers for disconnected nodes.
        """
        connected_components = list(nx.weakly_connected_components(self.graph))

        disconnected_nodes = []
        biggest_subgraph = max(connected_components, key=len)

        for component in connected_components:
            if component != biggest_subgraph:
                disconnected_nodes.extend(list(component))

        return disconnected_nodes

    async def extract_node(self, node_id: UUID) -> dict:
        """
        Retrieve data for a specific node based on its identifier.

        Parameters:
        -----------

            - node_id (UUID): The identifier of the node to retrieve.

        Returns:
        --------

            - dict: The data of the specified node, or None if not found.
        """
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]

        return None

    async def extract_nodes(self, node_ids: List[UUID]) -> List[dict]:
        """
        Retrieve data for multiple nodes based on their identifiers.

        Parameters:
        -----------

            - node_ids (List[UUID]): A list of node identifiers to retrieve data.

        Returns:
        --------

            - List[dict]: A list of data for each node identified that exists in the graph.
        """
        return [self.graph.nodes[node_id] for node_id in node_ids if self.graph.has_node(node_id)]

    async def get_predecessors(self, node_id: UUID, edge_label: str = None) -> list:
        """
        Retrieve the predecessor nodes of a specified node according to a specific edge label.

        Parameters:
        -----------

            - node_id (UUID): The identifier of the node for which to find predecessors.
            - edge_label (str): The label for the edges connecting to predecessors; if None, all
              predecessors are retrieved. (default None)

        Returns:
        --------

            - list: A list of predecessor nodes.
        """
        if self.graph.has_node(node_id):
            if edge_label is None:
                return [
                    self.graph.nodes[predecessor]
                    for predecessor in list(self.graph.predecessors(node_id))
                ]

            nodes = []

            for predecessor_id in list(self.graph.predecessors(node_id)):
                if self.graph.has_edge(predecessor_id, node_id, edge_label):
                    nodes.append(self.graph.nodes[predecessor_id])

            return nodes

    async def get_successors(self, node_id: UUID, edge_label: str = None) -> list:
        """
        Retrieve the successor nodes of a specified node according to a specific edge label.

        Parameters:
        -----------

            - node_id (UUID): The identifier of the node for which to find successors.
            - edge_label (str): The label for the edges connecting to successors; if None, all
              successors are retrieved. (default None)

        Returns:
        --------

            - list: A list of successor nodes.
        """
        if self.graph.has_node(node_id):
            if edge_label is None:
                return [
                    self.graph.nodes[successor]
                    for successor in list(self.graph.successors(node_id))
                ]

            nodes = []

            for successor_id in list(self.graph.successors(node_id)):
                if self.graph.has_edge(node_id, successor_id, edge_label):
                    nodes.append(self.graph.nodes[successor_id])

            return nodes

    async def get_neighbors(self, node_id: UUID) -> list:
        """
        Get the neighboring nodes of a specified node, including both predecessors and
        successors.

        Parameters:
        -----------

            - node_id (UUID): The identifier of the node whose neighbors are to be retrieved.

        Returns:
        --------

            - list: A list of neighboring nodes.
        """
        if not self.graph.has_node(node_id):
            return []

        predecessors, successors = await asyncio.gather(
            self.get_predecessors(node_id),
            self.get_successors(node_id),
        )

        neighbors = predecessors + successors

        return neighbors

    async def get_connections(self, node_id: UUID) -> list:
        """
        Get the connections of a specified node to its neighbors.

        Parameters:
        -----------

            - node_id (UUID): The identifier of the node for which to get connections.

        Returns:
        --------

            - list: A list of connections involving the specified node and its neighbors.
        """
        if not self.graph.has_node(node_id):
            return []

        node = self.graph.nodes[node_id]

        if "id" not in node:
            return []

        predecessors, successors = await asyncio.gather(
            self.get_predecessors(node_id),
            self.get_successors(node_id),
        )

        connections = []

        # Handle None values for predecessors and successors
        if predecessors is not None:
            for neighbor in predecessors:
                if "id" in neighbor:
                    edge_data = self.graph.get_edge_data(neighbor["id"], node["id"])
                    if edge_data is not None:
                        for edge_properties in edge_data.values():
                            connections.append((neighbor, edge_properties, node))

        if successors is not None:
            for neighbor in successors:
                if "id" in neighbor:
                    edge_data = self.graph.get_edge_data(node["id"], neighbor["id"])
                    if edge_data is not None:
                        for edge_properties in edge_data.values():
                            connections.append((node, edge_properties, neighbor))

        return connections

    async def remove_connection_to_predecessors_of(
        self, node_ids: list[UUID], edge_label: str
    ) -> None:
        """
        Remove connections to predecessors of specified nodes based on an edge label and persist
        changes.

        Parameters:
        -----------

            - node_ids (list[UUID]): A list of node identifiers whose predecessor connections
              need to be removed.
            - edge_label (str): The label of the edges to remove.
        """
        for node_id in node_ids:
            if self.graph.has_node(node_id):
                for predecessor_id in list(self.graph.predecessors(node_id)):
                    if self.graph.has_edge(predecessor_id, node_id, edge_label):
                        self.graph.remove_edge(predecessor_id, node_id, edge_label)

        await self.save_graph_to_file(self.filename)

    async def remove_connection_to_successors_of(
        self, node_ids: list[UUID], edge_label: str
    ) -> None:
        """
        Remove connections to successors of specified nodes based on an edge label and persist
        changes.

        Parameters:
        -----------

            - node_ids (list[UUID]): A list of node identifiers whose successor connections need
              to be removed.
            - edge_label (str): The label of the edges to remove.
        """
        for node_id in node_ids:
            if self.graph.has_node(node_id):
                for successor_id in list(self.graph.successors(node_id)):
                    if self.graph.has_edge(node_id, successor_id, edge_label):
                        self.graph.remove_edge(node_id, successor_id, edge_label)

        await self.save_graph_to_file(self.filename)

    async def create_empty_graph(self, file_path: str) -> None:
        """
        Initialize an empty graph and save it to a specified file path.

        Parameters:
        -----------

            - file_path (str): The file path where the empty graph should be saved.
        """
        self.graph = nx.MultiDiGraph()

        await self.save_graph_to_file(file_path)

    async def save_graph_to_file(self, file_path: str = None) -> None:
        """
        Save the graph data asynchronously to a specified file in JSON format.

        Parameters:
        -----------

            - file_path (str): The file path to save the graph data; if None, saves to the
              default filename. (default None)
        """
        if not file_path:
            file_path = self.filename

        graph_data = nx.readwrite.json_graph.node_link_data(self.graph, edges="links")

        file_dir_path = os.path.dirname(file_path)
        file_path = os.path.basename(file_path)

        file_storage = get_file_storage(file_dir_path)

        json_data = json.dumps(graph_data, cls=JSONEncoder)

        await file_storage.store(file_path, json_data, overwrite=True)

    async def load_graph_from_file(self, file_path: str = None):
        """
        Load graph data asynchronously from a specified file in JSON format.

        Parameters:
        -----------

            - file_path (str): The file path from which to load the graph data; if None, loads
              from the default filename. (default None)
        """
        if not file_path:
            file_path = self.filename
        try:
            file_dir_path = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)

            file_storage = get_file_storage(file_dir_path)

            if await file_storage.file_exists(file_name):
                async with file_storage.open(file_name, "r") as file:
                    graph_data = json.loads(file.read())
                    for node in graph_data["nodes"]:
                        try:
                            if not isinstance(node["id"], UUID):
                                try:
                                    node["id"] = UUID(node["id"])
                                except Exception:
                                    # If conversion fails, keep the original id
                                    pass
                        except Exception as e:
                            logger.error(e)
                            raise e

                        if isinstance(node.get("updated_at"), int):
                            node["updated_at"] = datetime.fromtimestamp(
                                node["updated_at"] / 1000, tz=timezone.utc
                            )
                        elif isinstance(node.get("updated_at"), str):
                            node["updated_at"] = datetime.strptime(
                                node["updated_at"], "%Y-%m-%dT%H:%M:%S.%f%z"
                            )

                    for edge in graph_data["links"]:
                        try:
                            if not isinstance(edge["source"], UUID):
                                source_id = parse_id(edge["source"])
                            else:
                                source_id = edge["source"]

                            if not isinstance(edge["target"], UUID):
                                target_id = parse_id(edge["target"])
                            else:
                                target_id = edge["target"]

                            edge["source"] = source_id
                            edge["target"] = target_id
                            edge["source_node_id"] = source_id
                            edge["target_node_id"] = target_id
                        except Exception as e:
                            logger.error(e)
                            raise e

                        if isinstance(
                            edge.get("updated_at"), int
                        ):  # Handle timestamp in milliseconds
                            edge["updated_at"] = datetime.fromtimestamp(
                                edge["updated_at"] / 1000, tz=timezone.utc
                            )
                        elif isinstance(edge.get("updated_at"), str):
                            edge["updated_at"] = datetime.strptime(
                                edge["updated_at"], "%Y-%m-%dT%H:%M:%S.%f%z"
                            )

                    self.graph = nx.readwrite.json_graph.node_link_graph(graph_data, edges="links")

                    for node_id, node_data in self.graph.nodes(data=True):
                        node_data["id"] = node_id
            else:
                # Log that the file does not exist and an empty graph is initialized
                logger.warning("File %s not found. Initializing an empty graph.", file_path)
                await self.create_empty_graph(file_path)

        except Exception:
            logger.error("Failed to load graph from file: %s", file_path)

            await self.create_empty_graph(file_path)

    async def delete_graph(self, file_path: str = None):
        """
        Delete the graph file from the filesystem asynchronously.

        Parameters:
        -----------

            - file_path (str): The file path of the graph to delete; if None, deletes the
              default graph file. (default None)
        """
        if file_path is None:
            file_path = (
                self.filename
            )  # Assuming self.filename is defined elsewhere and holds the default graph file path
        try:
            file_dir_path = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)

            file_storage = get_file_storage(file_dir_path)

            await file_storage.remove(file_name)

            self.graph = None
            logger.info("Graph deleted successfully.")
        except Exception as error:
            logger.error("Failed to delete graph: %s", error)
            raise error

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str]
    ) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]:
        """
        Obtain a subgraph based on specific node types and names. Not supported in this
        implementation.

        Parameters:
        -----------

            - node_type (Type[Any]): The type of nodes to include in the subgraph.
            - node_name (List[str]): A list of node names to filter by.
        """
        raise NodesetFilterNotSupportedError

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ):
        """
        Fetch nodes and relationships filtered by specified attributes.

        Parameters:
        -----------

            - attribute_filters (List[Dict[str, List[Union[str, int]]]]): A list of dictionaries
              defining attributes to filter on.

        Returns:
        --------

            A tuple containing filtered nodes and edges based on the specified attributes.
        """
        # Create filters for nodes based on the attribute filters
        where_clauses = []
        for attribute, values in attribute_filters[0].items():
            where_clauses.append((attribute, values))

        # Filter nodes
        filtered_nodes = [
            (node, data)
            for node, data in self.graph.nodes(data=True)
            if all(data.get(attr) in values for attr, values in where_clauses)
        ]

        # Filter edges where both source and target nodes satisfy the filters
        filtered_edges = [
            (source, target, data.get("relationship_type", "UNKNOWN"), data)
            for source, target, data in self.graph.edges(data=True)
            if (
                all(self.graph.nodes[source].get(attr) in values for attr, values in where_clauses)
                and all(
                    self.graph.nodes[target].get(attr) in values for attr, values in where_clauses
                )
            )
        ]

        return filtered_nodes, filtered_edges

    async def get_graph_metrics(self, include_optional=False):
        """
        Calculate various metrics related to the graph, optionally including optional metrics.

        Parameters:
        -----------

            - include_optional: Indicates whether optional metrics should be included in the
              calculation. (default False)

        Returns:
        --------

            A dictionary containing the calculated graph metrics.
        """
        graph = self.graph

        def _get_mean_degree(graph):
            degrees = [d for _, d in graph.degree()]
            return np.mean(degrees) if degrees else 0

        def _get_edge_density(graph):
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            num_possible_edges = num_nodes * (num_nodes - 1)
            edge_density = num_edges / num_possible_edges if num_possible_edges > 0 else 0
            return edge_density

        def _get_diameter(graph):
            try:
                return nx.diameter(nx.DiGraph(graph.to_undirected()))
            except Exception as e:
                logger.warning("Failed to calculate diameter: %s", e)
                return None

        def _get_avg_shortest_path_length(graph):
            try:
                return nx.average_shortest_path_length(nx.DiGraph(graph.to_undirected()))
            except Exception as e:
                logger.warning("Failed to calculate average shortest path length: %s", e)
                return None

        def _get_avg_clustering(graph):
            try:
                return nx.average_clustering(nx.DiGraph(graph.to_undirected()))
            except Exception as e:
                logger.warning("Failed to calculate clustering coefficient: %s", e)
                return None

        mandatory_metrics = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "mean_degree": _get_mean_degree(graph),
            "edge_density": _get_edge_density(graph),
            "num_connected_components": nx.number_weakly_connected_components(graph),
            "sizes_of_connected_components": [
                len(c) for c in nx.weakly_connected_components(graph)
            ],
        }

        if include_optional:
            optional_metrics = {
                "num_selfloops": sum(1 for u, v in graph.edges() if u == v),
                "diameter": _get_diameter(graph),
                "avg_shortest_path_length": _get_avg_shortest_path_length(graph),
                "avg_clustering": _get_avg_clustering(graph),
            }
        else:
            optional_metrics = {
                "num_selfloops": -1,
                "diameter": -1,
                "avg_shortest_path_length": -1,
                "avg_clustering": -1,
            }

        return mandatory_metrics | optional_metrics

    async def get_document_subgraph(self, content_hash: str):
        """
        Retrieve all relevant nodes when a document is being deleted, including chunks and
        orphaned entities.

        Parameters:
        -----------

            - content_hash (str): The hash identifying the content of the document to fetch
              related nodes for.

        Returns:
        --------

            A dictionary containing the document, its chunks, orphan entities, made from nodes,
            and orphan types.
        """
        # Ensure graph is loaded
        if self.graph is None:
            await self.load_graph_from_file()

        # Find the document node by looking for content_hash in the name field
        document = None
        document_node_id = None
        for node_id, attrs in self.graph.nodes(data=True):
            if (
                attrs.get("type") in ["TextDocument", "PdfDocument"]
                and attrs.get("name") == f"text_{content_hash}"
            ):
                document = {"id": str(node_id), **attrs}  # Convert UUID to string for consistency
                document_node_id = node_id  # Keep the original UUID
                break

        if not document:
            return None

        # Find chunks connected via is_part_of (chunks point TO document)
        chunks = []
        for source, target, edge_data in self.graph.in_edges(document_node_id, data=True):
            if edge_data.get("relationship_name") == "is_part_of":
                chunks.append({"id": source, **self.graph.nodes[source]})  # Keep as UUID object

        # Find entities connected to chunks (chunks point TO entities via contains)
        entities = []
        for chunk in chunks:
            chunk_id = chunk["id"]  # Already a UUID object
            for source, target, edge_data in self.graph.out_edges(chunk_id, data=True):
                if edge_data.get("relationship_name") == "contains":
                    entities.append(
                        {"id": target, **self.graph.nodes[target]}
                    )  # Keep as UUID object

        # Find orphaned entities (entities only connected to chunks we're deleting)
        orphan_entities = []
        for entity in entities:
            entity_id = entity["id"]  # Already a UUID object
            # Get all chunks that contain this entity
            containing_chunks = []
            for source, target, edge_data in self.graph.in_edges(entity_id, data=True):
                if edge_data.get("relationship_name") == "contains":
                    containing_chunks.append(source)  # Keep as UUID object

            # Check if all containing chunks are in our chunks list
            chunk_ids = [chunk["id"] for chunk in chunks]
            if containing_chunks and all(c in chunk_ids for c in containing_chunks):
                orphan_entities.append(entity)

        # Find orphaned entity types
        orphan_types = []
        seen_types = set()  # Track seen types to avoid duplicates
        for entity in orphan_entities:
            entity_id = entity["id"]  # Already a UUID object
            for _, target, edge_data in self.graph.out_edges(entity_id, data=True):
                if edge_data.get("relationship_name") in ["is_a", "instance_of"]:
                    # Check if this type is only connected to entities we're deleting
                    type_node = self.graph.nodes[target]
                    if type_node.get("type") == "EntityType" and target not in seen_types:
                        is_orphaned = True
                        # Get all incoming edges to this type node
                        for source, _, edge_data in self.graph.in_edges(target, data=True):
                            if edge_data.get("relationship_name") in ["is_a", "instance_of"]:
                                # Check if the source entity is not in our orphan_entities list
                                if source not in [e["id"] for e in orphan_entities]:
                                    is_orphaned = False
                                    break
                        if is_orphaned:
                            orphan_types.append({"id": target, **type_node})  # Keep as UUID object
                            seen_types.add(target)  # Mark as seen

        # Find nodes connected via made_from (chunks point TO summaries)
        made_from_nodes = []
        for chunk in chunks:
            chunk_id = chunk["id"]  # Already a UUID object
            for source, target, edge_data in self.graph.in_edges(chunk_id, data=True):
                if edge_data.get("relationship_name") == "made_from":
                    made_from_nodes.append(
                        {"id": source, **self.graph.nodes[source]}
                    )  # Keep as UUID object

        # Return UUIDs directly without string conversion
        return {
            "document": [{"id": document["id"], **{k: v for k, v in document.items() if k != "id"}}]
            if document
            else [],
            "chunks": [
                {"id": chunk["id"], **{k: v for k, v in chunk.items() if k != "id"}}
                for chunk in chunks
            ],
            "orphan_entities": [
                {"id": entity["id"], **{k: v for k, v in entity.items() if k != "id"}}
                for entity in orphan_entities
            ],
            "made_from_nodes": [
                {"id": node["id"], **{k: v for k, v in node.items() if k != "id"}}
                for node in made_from_nodes
            ],
            "orphan_types": [
                {"id": type_node["id"], **{k: v for k, v in type_node.items() if k != "id"}}
                for type_node in orphan_types
            ],
        }

    async def get_degree_one_nodes(self, node_type: str):
        """
        Retrieve nodes that have only a single connection, filtered by node type.

        Parameters:
        -----------

            - node_type (str): Type of nodes to filter by ('Entity' or 'EntityType').

        Returns:
        --------

            A list of nodes that have a single connection of the specified type.
        """
        if not node_type or node_type not in ["Entity", "EntityType"]:
            raise ValueError("node_type must be either 'Entity' or 'EntityType'")

        nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("type") == node_type:
                # Count both incoming and outgoing edges
                degree = self.graph.degree(node_id)
                if degree == 1:
                    nodes.append(node_data)
        return nodes

    async def get_node(self, node_id: UUID) -> dict:
        """
        Retrieve the details of a specific node identified by its identifier.

        Parameters:
        -----------

            - node_id (UUID): The identifier of the node to retrieval.

        Returns:
        --------

            - dict: The data of the specified node if found, otherwise None.
        """
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return None

    async def get_nodes(self, node_ids: List[UUID] = None) -> List[dict]:
        """
        Retrieve data for multiple nodes by their identifiers, or all nodes if no identifiers
        are provided.

        Parameters:
        -----------

            - node_ids (List[UUID]): List of node identifiers to fetch data for; if None,
              retrieves all nodes in the graph. (default None)

        Returns:
        --------

            - List[dict]: A list of node data for each found node.
        """
        if node_ids is None:
            return [{"id": node_id, **data} for node_id, data in self.graph.nodes(data=True)]
        return [
            {"id": node_id, **self.graph.nodes[node_id]}
            for node_id in node_ids
            if self.graph.has_node(node_id)
        ]
