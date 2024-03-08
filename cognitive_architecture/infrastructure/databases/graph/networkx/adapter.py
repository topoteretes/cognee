import pickle
from datetime import datetime
import aiofiles
import networkx as nx
from cognitive_architecture.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
import logging

class NetworXDB(GraphDBInterface):
    def __init__(self, filename="cognee_graph.pkl"):
        self.filename = filename
        self.graph = nx.MultiDiGraph()


    async def save_graph(self, path: str):
        """Asynchronously save the graph to a file."""
        if path is not None:
            path = self.filename
        try:
            async with aiofiles.open(path, "wb") as f:
                await f.write(pickle.dumps(self.graph))
            logging.info("Graph saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save graph: {e}")

    async def load_graph(self, path: str):
        if path is not None:
            path = self.filename
        try:
            async with aiofiles.open(path, "rb") as f:
                data = await f.read()
                self.graph = pickle.loads(data)
            logging.info("Graph loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load graph: {e}")

    async def delete_graph(self, path: str):
        if path is not None:
            path = self.filename
        try:
            async with aiofiles.open(path, "wb") as f:
                await f.write(pickle.dumps(self.graph))
            logging.info("Graph deleted successfully.")
        except Exception as e:
            logging.error(f"Failed to delete graph: {e}")

    async def create(self, user_id, custom_user_properties=None, required_layers=None, default_fields=None):
        """Asynchronously create or update a user content graph based on given parameters."""
        # Assume required_layers is a dictionary-like object; use more robust validation in production
        category_name = required_layers['name']
        subgroup_names = [subgroup['name'] for subgroup in required_layers['cognitive_subgroups']]

        # Construct the additional_categories structure
        additional_categories = {category_name: subgroup_names}

        # Define default fields for all nodes if not provided
        if default_fields is None:
            default_fields = {
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        # Merge custom user properties with default fields; custom properties take precedence
        user_properties = {**default_fields, **(custom_user_properties or {})}

        # Default content categories and update with any additional categories provided
        content_categories = {
            "Temporal": ["Historical events", "Schedules and timelines"],
            "Positional": ["Geographical locations", "Spatial data"],
            "Propositions": ["Hypotheses and theories", "Claims and arguments"],
            "Personalization": ["User preferences", "User information"]
        }
        content_categories.update(additional_categories)

        # Ensure the user node exists with properties
        self.graph.add_node(user_id, **user_properties, exist=True)

        # Add or update content category nodes and their edges
        for category, subclasses in content_categories.items():
            category_properties = {**default_fields, 'type': 'category'}
            self.graph.add_node(category, **category_properties, exist=True)
            self.graph.add_edge(user_id, category, relationship='created')

            # Add or update subclass nodes and their edges
            for subclass in subclasses:
                subclass_node_id = f"{category}:{subclass}"
                subclass_properties = {**default_fields, 'type': 'subclass', 'content': subclass}
                self.graph.add_node(subclass_node_id, **subclass_properties, exist=True)
                self.graph.add_edge(category, subclass_node_id, relationship='includes')

        # Save the graph asynchronously after modifications
        await self.save_graph()