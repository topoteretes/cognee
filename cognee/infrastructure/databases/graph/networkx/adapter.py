"""Adapter for NetworkX graph database."""

import os
import json
import logging
from typing import Dict, Any, List
import aiofiles
import networkx as nx
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

logger = logging.getLogger(__name__)

class NetworkXAdapter(GraphDBInterface):
    _instance = None  # Class variable to store the singleton instance

    def __new__(cls, filename):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.filename = filename
        return cls._instance

    def __init__(self, filename = "cognee_graph.pkl"):
        self.filename = filename
        self.graph = nx.MultiDiGraph()

    async def graph(self):
        return self.graph

    async def add_node(
        self,
        node_id: str,
        node_properties,
    ) -> None:
        if not self.graph.has_node(id):
            # current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.graph.add_node(node_id, **node_properties)
            await self.save_graph_to_file(self.filename)

    async def add_nodes(
        self,
        nodes: List[tuple[str, dict]],
    ) -> None:
        self.graph.add_nodes_from(nodes)
        await self.save_graph_to_file(self.filename)
    
    async def add_edge(
        self,
        from_node: str,
        to_node: str,
        relationship_name: str,
        edge_properties: Dict[str, Any] = None,
    ) -> None:
        self.graph.add_edge(from_node, to_node, key = relationship_name, **(edge_properties if edge_properties else {}))
        await self.save_graph_to_file(self.filename)

    async def add_edges(
        self,
        edges: tuple[str, str, dict],
    ) -> None:
        self.graph.add_edges_from(edges)
        await self.save_graph_to_file(self.filename)

    async def delete_node(self, node_id: str) -> None:
        """Asynchronously delete a node from the graph if it exists."""
        if self.graph.has_node(id):
            self.graph.remove_node(id)
            await self.save_graph_to_file(self.filename)

    async def extract_node_description(self, node_id: str) -> Dict[str, Any]:
        descriptions = []

        if self.graph.has_node(node_id):
            # Get the attributes of the node
            for neighbor in self.graph.neighbors(node_id):
                # Get the attributes of the neighboring node
                attributes = self.graph.nodes[neighbor]

                # Ensure all required attributes are present before extracting description
                if all(key in attributes for key in
                       ["description", "unique_id", "layer_uuid", "layer_decomposition_uuid"]):
                    descriptions.append({
                        "node_id": attributes["unique_id"],
                        "description": attributes["description"],
                        "layer_uuid": attributes["layer_uuid"],
                        "layer_decomposition_uuid": attributes["layer_decomposition_uuid"]
                    })

        return descriptions


    async def extract_node(self, node_id: str) -> dict:
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]

        return None


    async def save_graph_to_file(self, file_path: str=None) -> None:
        """Asynchronously save the graph to a file in JSON format."""
        if not file_path:
            file_path = self.filename

        graph_data = nx.readwrite.json_graph.node_link_data(self.graph)

        async with aiofiles.open(file_path, "w") as file:
            await file.write(json.dumps(graph_data))

    async def load_graph_from_file(self, file_path: str = None):
        """Asynchronously load the graph from a file in JSON format."""
        if not file_path:
            file_path = self.filename
        try:
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, "r") as file:
                    graph_data = json.loads(await file.read())
                    self.graph = nx.readwrite.json_graph.node_link_graph(graph_data)
                    return self.graph
            else:
                # Log that the file does not exist and an empty graph is initialized
                logger.warning("File %s not found. Initializing an empty graph.", file_path)
                self.graph = nx.MultiDiGraph()  # Use MultiDiGraph to keep it consistent with __init__
                return self.graph
        except Exception as error:
            logger.error("Failed to load graph from file: %s", file_path)
            # Initialize an empty graph in case of error
            self.graph = nx.MultiDiGraph()
            return self.graph

    async def delete_graph_from_file(self, path: str = None):
        """Asynchronously delete the graph file from the filesystem."""
        if path is None:
            path = self.filename  # Assuming self.filename is defined elsewhere and holds the default graph file path
        try:
            await aiofiles.os.remove(path)
            logger.info("Graph deleted successfully.")
        except Exception as error:
            logger.error("Failed to delete graph: %s", error)
