""" Here we update semantic graph with content that classifier produced"""
import uuid
from datetime import datetime
from enum import Enum, auto
from typing import Type, Optional, Any
from pydantic import BaseModel
from cognitive_architecture.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognitive_architecture.shared.data_models import GraphDBType, DefaultGraphModel, Document, DocumentType, Category, Relationship, UserProperties, UserLocation


def add_propositions(G, category_name, subclass_content, layer_description, new_data, layer_uuid,
                             layer_decomposition_uuid):
    """ Add nodes and edges to the graph for the given LLM knowledge graph and the layer"""

    # Find the node ID for the subclass within the category
    G.load_graph_from_file()
    G = graph_client.graph
    subclass_node_id = None
    for node, data in G.nodes(data=True):
        if subclass_content in node:
            subclass_node_id = node

            print(subclass_node_id)

    if not subclass_node_id:
        print(f"Subclass '{subclass_content}' under category '{category_name}' not found in the graph.")
        return G

    # Mapping from old node IDs to new node IDs
    node_id_mapping = {}

    # Add nodes from the Pydantic object
    for node in new_data.nodes:
        unique_node_id = uuid.uuid4()
        new_node_id = f"{node.description} - {str(layer_uuid)}  - {str(layer_decomposition_uuid)} - {str(unique_node_id)}"
        G.add_node(new_node_id,
                   created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   description=node.description,
                   category=node.category,
                   memory_type=node.memory_type,
                   layer_uuid=str(layer_uuid),
                   layer_description=str(layer_description),
                   layer_decomposition_uuid=str(layer_decomposition_uuid),
                   id=str(unique_node_id),
                   type='detail')
        G.add_edge(subclass_node_id, new_node_id, relationship='detail')

        # Store the mapping from old node ID to new node ID
        node_id_mapping[node.id] = new_node_id

    # Add edges from the Pydantic object using the new node IDs
    for edge in new_data.edges:
        # Use the mapping to get the new node IDs
        source_node_id = node_id_mapping.get(edge.source)
        target_node_id = node_id_mapping.get(edge.target)

        if source_node_id and target_node_id:
            G.add_edge(source_node_id, target_node_id, description=edge.description, relationship='relation')
        else:
            print(f"Could not find mapping for edge from {edge.source} to {edge.target}")

    return G




if __name__ == "__main__":
    import asyncio

    # Assuming all necessary imports and GraphDBType, get_graph_client, Document, DocumentType, etc. are defined

    # Initialize the graph client
    graph_client = get_graph_client(GraphDBType.NETWORKX)
    G = asyncio.run(add_propositions(graph_client, 'category_name', 'subclass_content', 'layer_description', 'new_data', 'layer_uuid',
                             'layer_decomposition_uuid'))














if __name__ == "__main__":
    import asyncio

    # Assuming all necessary imports and GraphDBType, get_graph_client, Document, DocumentType, etc. are defined

    # Initialize the graph client
    graph_client = get_graph_client(GraphDBType.NETWORKX)
