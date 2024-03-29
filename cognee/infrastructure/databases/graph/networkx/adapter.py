"""Adapter for NetworkX graph database."""

import json
import os
import pickle
from datetime import datetime
from typing import Optional, Dict, Any
import aiofiles.os
import aiofiles
import networkx as nx
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
import logging

class NetworXAdapter(GraphDBInterface):
    _instance = None  # Class variable to store the singleton instance


    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super(NetworXAdapter, cls).__new__(cls)
    #     return cls._instance
    def __init__(self, filename="cognee_graph.pkl"):
        # self.load_graph_from_file()
        self.filename = filename
        self.graph = nx.MultiDiGraph()


    @classmethod
    async def async_create(cls, filename="cognee_graph.pkl"):
        instance = cls()
        instance.filename = filename
        await instance.load_graph_from_file()
        return instance
    async def graph(self):
        return self.graph
        # G = await client.load_graph_from_file()
        # if G is None:
        #     G = client.graph  # Directly access the graph attribute without calling it
        # return G


    async def add_node(self, id: str, **kwargs) -> None:
        """Asynchronously add a node to the graph if it doesn't already exist, with given properties."""
        if not self.graph.has_node(id):
            self.graph.add_node(id, **kwargs)
            await self.save_graph_to_file(self.filename)

    async def add_edge(self, from_node: str, to_node: str, **kwargs ) -> None:
        """Asynchronously add an edge between two nodes with optional properties."""
        # properties = properties or {}
        self.graph.add_edge(from_node, to_node, **kwargs)
        await self.save_graph_to_file(self.filename)

    async def delete_node(self, id: str) -> None:
        """Asynchronously delete a node from the graph if it exists."""
        if self.graph.has_node(id):
            self.graph.remove_node(id)
            await self.save_graph_to_file(self.filename)


    async def save_graph_to_file(self, file_path: str=None) -> None:
        """Asynchronously save the graph to a file in JSON format."""
        if not file_path:
            file_path = self.filename
        graph_data = nx.readwrite.json_graph.node_link_data(self.graph)
        async with aiofiles.open(file_path, 'w') as file:
            await file.write(json.dumps(graph_data))

    async def load_graph_from_file(self, file_path: str = None):
        """Asynchronously load the graph from a file in JSON format."""
        if not file_path:
            file_path = self.filename
        try:
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, 'r') as file:
                    graph_data = json.loads(await file.read())
                    self.graph = nx.readwrite.json_graph.node_link_graph(graph_data)
                    return self.graph
            else:
                # Log that the file does not exist and an empty graph is initialized
                logging.warning(f"File {file_path} not found. Initializing an empty graph.")
                self.graph = nx.MultiDiGraph()  # Use MultiDiGraph to keep it consistent with __init__
                return self.graph
        except Exception as e:
            logging.error(f"Failed to load graph from {file_path}: {e}")
            # Consider initializing an empty graph in case of error
            self.graph = nx.MultiDiGraph()
            return self.graph

    async def delete_graph_from_file(self, path: str = None):
        """Asynchronously delete the graph file from the filesystem."""
        if path is None:
            path = self.filename  # Assuming self.filename is defined elsewhere and holds the default graph file path
        try:
            await aiofiles.os.remove(path)  # Asynchronously remove the file
            logging.info("Graph deleted successfully.")
        except Exception as e:
            logging.error(f"Failed to delete graph: {e}")

    # async def create(self, user_id, custom_user_properties=None, required_layers=None, default_fields=None, existing_graph=None):
    #     """Asynchronously create or update a user content graph based on given parameters."""
    #     # Assume required_layers is a dictionary-like object; use more robust validation in production
    #     category_name = required_layers['data_type']
    #     subgroup_names = [required_layers['layer_name']]
    #
    #     # Construct the additional_categories structure
    #     additional_categories = {category_name: subgroup_names}
    #
    #     # Define default fields for all nodes if not provided
    #     if default_fields is None:
    #         default_fields = {
    #             'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #             'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         }
    #
    #     # Merge custom user properties with default fields; custom properties take precedence
    #     user_properties = {**default_fields, **(custom_user_properties or {})}
    #
    #     # Default content categories and update with any additional categories provided
    #     content_categories = {
    #         "Temporal": ["Historical events", "Schedules and timelines"],
    #         "Positional": ["Geographical locations", "Spatial data"],
    #         "Propositions": ["Hypotheses and theories", "Claims and arguments"],
    #         "Personalization": ["User preferences", "User information"]
    #     }
    #
    #     content_categories = {
    #         "Temporal": ["Historical events", "Schedules and timelines"],
    #         "Positional": ["Geographical locations", "Spatial data"],
    #         "Propositions": ["Hypotheses and theories", "Claims and arguments"],
    #         "Personalization": ["User preferences", "User information"]
    #     }
    #
    #     # Update content categories with any additional categories provided
    #     if additional_categories:
    #         content_categories.update(additional_categories)
    #
    #     G = existing_graph if existing_graph else self.graph
    #
    #     # Check if the user node already exists, if not, add the user node with properties
    #     if not G.has_node(user_id):
    #         G.add_node(user_id, **user_properties)
    #
    #     # Add or update content category nodes and their edges
    #     for category, subclasses in content_categories.items():
    #         category_properties = {**default_fields, 'type': 'category'}
    #
    #         # Add or update the category node
    #         if not G.has_node(category):
    #             G.add_node(category, **category_properties)
    #             G.add_edge(user_id, category, relationship='created')
    #
    #         # Add or update subclass nodes and their edges
    #         for subclass in subclasses:
    #             # Using both category and subclass names to ensure uniqueness within categories
    #             subclass_node_id = f"{category}:{subclass}"
    #
    #             # Check if subclass node exists before adding, based on node content
    #             if not any(subclass == data.get('content') for _, data in G.nodes(data=True)):
    #                 subclass_properties = {**default_fields, 'type': 'subclass', 'content': subclass}
    #                 G.add_node(subclass_node_id, **subclass_properties)
    #                 G.add_edge(category, subclass_node_id, relationship='includes')
    #
    #     return G
        # content_categories.update(additional_categories)
        #
        # # Ensure the user node exists with properties
        # self.graph.add_node(user_id, **user_properties, exist=True)
        #
        # # Add or update content category nodes and their edges
        # for category, subclasses in content_categories.items():
        #     category_properties = {**default_fields, 'type': 'category'}
        #     self.graph.add_node(category, **category_properties, exist=True)
        #     self.graph.add_edge(user_id, category, relationship='created')
        #
        #     # Add or update subclass nodes and their edges
        #     for subclass in subclasses:
        #         subclass_node_id = f"{category}:{subclass}"
        #         subclass_properties = {**default_fields, 'type': 'subclass', 'content': subclass}
        #         self.graph.add_node(subclass_node_id, **subclass_properties, exist=True)
        #         self.graph.add_edge(category, subclass_node_id, relationship='includes')
        #
        # # Save the graph asynchronously after modifications
        # # await self.save_graph()
        #
        # return self.graph