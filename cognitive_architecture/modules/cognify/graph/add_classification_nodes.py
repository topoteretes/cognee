""" Here we update semantic graph with content that classifier produced"""
from datetime import datetime
from enum import Enum, auto
from typing import Type, Optional, Any
from pydantic import BaseModel
from cognitive_architecture.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognitive_architecture.shared.data_models import GraphDBType, DefaultGraphModel, Document, DocumentType, Category, Relationship, UserProperties, UserLocation


def add_classification_nodes(G, id, classification_data):
    context = classification_data['context_name']
    layer = classification_data['layer_name']

    # Create the layer classification node ID using the context_name
    layer_classification_node_id = f'LLM_LAYER_CLASSIFICATION:{context}:{id}'

    # Add the node to the graph, unpacking the node data from the dictionary
    G.add_node(layer_classification_node_id, **classification_data)

    # Link this node to the corresponding document node
    G.add_edge(id, layer_classification_node_id, relationship='classified_as')

    # Create the detailed classification node ID using the context_name
    detailed_classification_node_id = f'LLM_CLASSIFICATION:LAYER:{layer}:{id}'

    # Add the detailed classification node, reusing the same node data
    G.add_node(detailed_classification_node_id, **classification_data)

    # Link the detailed classification node to the layer classification node
    G.add_edge(layer_classification_node_id, detailed_classification_node_id, relationship='contains_analysis')
    return G





if __name__ == "__main__":
    import asyncio

    # Assuming all necessary imports and GraphDBType, get_graph_client, Document, DocumentType, etc. are defined

    # Initialize the graph client
    graph_client = get_graph_client(GraphDBType.NETWORKX)


    G = asyncio.run(add_classification_nodes(graph_client, 'document_id', {'data_type': 'text',
 'context_name': 'TEXT',
 'layer_name': 'Articles, essays, and reports'}))