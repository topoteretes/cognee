"""Adapter for NetworkX graph database."""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List
import aiofiles
import aiofiles.os as aiofiles_os
import networkx as nx
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

logger = logging.getLogger("NetworkXAdapter")

class NetworkXAdapter(GraphDBInterface):
    _instance = None
    graph = None # Class variable to store the singleton instance

    def __new__(cls, filename):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.filename = filename
        return cls._instance

    def __init__(self, filename = "cognee_graph.pkl"):
        self.filename = filename


    async def has_node(self, node_id: str) -> bool:
        return self.graph.has_node(node_id)

    async def add_node(
        self,
        node_id: str,
        node_properties,
    ) -> None:
        if not self.graph.has_node(id):
            self.graph.add_node(node_id, **node_properties)
            await self.save_graph_to_file(self.filename)

    async def add_nodes(
        self,
        nodes: List[tuple[str, dict]],
    ) -> None:
        self.graph.add_nodes_from(nodes)
        await self.save_graph_to_file(self.filename)

    async def get_graph(self):
        return self.graph

    async def has_edge(self, from_node: str, to_node: str, edge_label: str) -> bool:
        return self.graph.has_edge(from_node, to_node, key = edge_label)

    async def has_edges(self, edges):
        result = []

        for (from_node, to_node, edge_label) in edges:
            if await self.has_edge(from_node, to_node, edge_label):
                result.append((from_node, to_node, edge_label))

        return result

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
        edges: tuple[str, str, str, dict],
    ) -> None:
        self.graph.add_edges_from(edges)
        await self.save_graph_to_file(self.filename)

    async def get_edges(self, node_id: str):
        return list(self.graph.in_edges(node_id, data = True)) + list(self.graph.out_edges(node_id, data = True))

    async def delete_node(self, node_id: str) -> None:
        """Asynchronously delete a node from the graph if it exists."""
        if self.graph.has_node(id):
            self.graph.remove_node(id)
            await self.save_graph_to_file(self.filename)

    async def delete_nodes(self, node_ids: List[str]) -> None:
        self.graph.remove_nodes_from(node_ids)
        await self.save_graph_to_file(self.filename)

    async def get_disconnected_nodes(self) -> List[str]:
        connected_components = list(nx.weakly_connected_components(self.graph))

        disconnected_nodes = []
        biggest_subgraph = max(connected_components, key = len)

        for component in connected_components:
            if component != biggest_subgraph:
                disconnected_nodes.extend(list(component))

        return disconnected_nodes

    async def extract_node_description(self, node_id: str) -> Dict[str, Any]:
        descriptions = []

        if self.graph.has_node(node_id):
            # Get the attributes of the node
            for neighbor in self.graph.neighbors(node_id):
                # Get the attributes of the neighboring node
                attributes = self.graph.nodes[neighbor]

                # Ensure all required attributes are present before extracting description
                if all(key in attributes for key in ["id", "layer_id", "description"]):
                    descriptions.append({
                        "id": attributes["id"],
                        "layer_id": attributes["layer_id"],
                        "description": attributes["description"],
                    })

        return descriptions

    async def get_layer_nodes(self):
        layer_nodes = []

        for _, data in self.graph.nodes(data = True):
            if "layer_id" in data:
                layer_nodes.append(data)

        return layer_nodes

    async def extract_node(self, node_id: str) -> dict:
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]

        return None

    async def extract_nodes(self, node_ids: List[str]) -> List[dict]:
        return [self.graph.nodes[node_id] for node_id in node_ids if self.graph.has_node(node_id)]

    async def get_predecessors(self, node_id: str, edge_label: str = None) -> list:
        if self.graph.has_node(node_id):
            if edge_label is None:
                return [
                    self.graph.nodes[predecessor] for predecessor \
                        in list(self.graph.predecessors(node_id))
                ]

            nodes = []

            for predecessor_id in list(self.graph.predecessors(node_id)):
                if self.graph.has_edge(predecessor_id, node_id, edge_label):
                    nodes.append(self.graph.nodes[predecessor_id])

            return nodes

    async def get_successors(self, node_id: str, edge_label: str = None) -> list:
        if self.graph.has_node(node_id):
            if edge_label is None:
                return [
                    self.graph.nodes[successor] for successor \
                        in list(self.graph.successors(node_id))
                ]

            nodes = []

            for successor_id in list(self.graph.successors(node_id)):
                if self.graph.has_edge(node_id, successor_id, edge_label):
                    nodes.append(self.graph.nodes[successor_id])

            return nodes

    async def get_neighbours(self, node_id: str) -> list:
        if not self.graph.has_node(node_id):
            return []

        predecessors, successors = await asyncio.gather(
            self.get_predecessors(node_id),
            self.get_successors(node_id),
        )

        neighbours = predecessors + successors

        return neighbours

    async def get_connections(self, node_id: str) -> list:
        if not self.graph.has_node(node_id):
            return []

        node = self.graph.nodes[node_id]

        if "uuid" not in node:
            return []

        predecessors, successors = await asyncio.gather(
            self.get_predecessors(node_id),
            self.get_successors(node_id),
        )

        connections = []

        for neighbor in predecessors:
            if "uuid" in neighbor:
                edge_data = self.graph.get_edge_data(neighbor["uuid"], node["uuid"])
                for edge_properties in edge_data.values():
                    connections.append((neighbor, edge_properties, node))

        for neighbor in successors:
            if "uuid" in neighbor:
                edge_data = self.graph.get_edge_data(node["uuid"], neighbor["uuid"])
                for edge_properties in edge_data.values():
                    connections.append((node, edge_properties, neighbor))

        return connections

    async def remove_connection_to_predecessors_of(self, node_ids: list[str], edge_label: str) -> None:
        for node_id in node_ids:
            if self.graph.has_node(node_id):
                for predecessor_id in list(self.graph.predecessors(node_id)):
                    if self.graph.has_edge(predecessor_id, node_id, edge_label):
                        self.graph.remove_edge(predecessor_id, node_id, edge_label)

        await self.save_graph_to_file(self.filename)

    async def remove_connection_to_successors_of(self, node_ids: list[str], edge_label: str) -> None:
        for node_id in node_ids:
            if self.graph.has_node(node_id):
                for successor_id in list(self.graph.successors(node_id)):
                    if self.graph.has_edge(node_id, successor_id, edge_label):
                        self.graph.remove_edge(node_id, successor_id, edge_label)

        await self.save_graph_to_file(self.filename)

    async def save_graph_to_file(self, file_path: str=None) -> None:
        """Asynchronously save the graph to a file in JSON format."""
        if not file_path:
            file_path = self.filename

        graph_data = nx.readwrite.json_graph.node_link_data(self.graph)

        async with aiofiles.open(file_path, "w") as file:
            await file.write(json.dumps(graph_data))


    async def load_graph_from_file(self, file_path: str = None):
        """Asynchronously load the graph from a file in JSON format."""
        if file_path == self.filename:
            return

        if not file_path:
            file_path = self.filename
        try:
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, "r") as file:
                    graph_data = json.loads(await file.read())
                    self.graph = nx.readwrite.json_graph.node_link_graph(graph_data)
            else:
                # Log that the file does not exist and an empty graph is initialized
                logger.warning("File %s not found. Initializing an empty graph.", file_path)
                self.graph = nx.MultiDiGraph()  # Use MultiDiGraph to keep it consistent with __init__

                file_dir = os.path.dirname(file_path)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir, exist_ok = True)

                await self.save_graph_to_file(file_path)
        except Exception:
            logger.error("Failed to load graph from file: %s", file_path)
            # Initialize an empty graph in case of error
            self.graph = nx.MultiDiGraph()

            file_dir = os.path.dirname(file_path)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir, exist_ok = True)

            await self.save_graph_to_file(file_path)

    async def delete_graph(self, file_path: str = None):
        """Asynchronously delete the graph file from the filesystem."""
        if file_path is None:
            file_path = self.filename  # Assuming self.filename is defined elsewhere and holds the default graph file path
        try:
            if os.path.exists(file_path):
                await aiofiles_os.remove(file_path)

            self.graph = None
            logger.info("Graph deleted successfully.")
        except Exception as error:
            logger.error("Failed to delete graph: %s", error)
