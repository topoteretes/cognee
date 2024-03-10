""" Here we update semantic graph with content that classifier produced"""
import uuid
import json
from datetime import datetime
from enum import Enum, auto
from typing import Type, Optional, Any
from pydantic import BaseModel
from cognitive_architecture.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognitive_architecture.modules.cognify.llm.content_to_propositions import generate_graph
from cognitive_architecture.shared.data_models import GraphDBType, DefaultGraphModel, Document, DocumentType, Category, \
    Relationship, UserProperties, UserLocation, KnowledgeGraph, TextSubclass, TextContent, DefaultContentPrediction


async def add_propositions(G, category_name, subclass_content, layer_description, new_data, layer_uuid,
                             layer_decomposition_uuid):
    """ Add nodes and edges to the graph for the given LLM knowledge graph and the layer"""

    # Find the node ID for the subclass within the category
    await G.load_graph_from_file()

    subclass_node_id = None
    for node, data in G.graph.nodes(data=True):
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
        await G.add_node(new_node_id,
                   created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   description=node.description,
                   category=node.category,
                   memory_type=node.memory_type,
                   layer_uuid=str(layer_uuid),
                   layer_description=str(layer_description),
                   layer_decomposition_uuid=str(layer_decomposition_uuid),
                   unique_id=str(unique_node_id),
                   type='detail')
        await G.add_edge(subclass_node_id, new_node_id, relationship='detail')

        # Store the mapping from old node ID to new node ID
        node_id_mapping[node.id] = new_node_id

    # Add edges from the Pydantic object using the new node IDs
    for edge in new_data.edges:
        # Use the mapping to get the new node IDs
        source_node_id = node_id_mapping.get(edge.source)
        target_node_id = node_id_mapping.get(edge.target)

        if source_node_id and target_node_id:
            await G.add_edge(source_node_id, target_node_id, description=edge.description, relationship='relation')
        else:
            print(f"Could not find mapping for edge from {edge.source} to {edge.target}")

    return G

async def append_to_graph(layer_graphs, required_layers, G):
    # Generate a UUID for the overall layer
    layer_uuid = uuid.uuid4()

    # Extract category name from required_layers data
    category_name = required_layers.dict()['label']['type']

    # Extract subgroup name from required_layers data
    # Assuming there's always at least one subclass and we're taking the first
    subgroup_name = required_layers.dict()['label']['subclass'][0].value  # Access the value of the enum

    for layer_ind in layer_graphs:

        for layer_json, knowledge_graph in layer_ind.items():
            # Decode the JSON key to get the layer description
            layer_description = json.loads(layer_json)

            # Generate a UUID for this particular layer decomposition
            layer_decomposition_uuid = uuid.uuid4()

            # Assuming append_data_to_graph is defined elsewhere and appends data to G
            # You would pass relevant information from knowledge_graph along with other details to this function
            F = await add_propositions(G, category_name, subgroup_name, layer_description, knowledge_graph,
                                     layer_uuid, layer_decomposition_uuid)

            # Print updated graph for verification (assuming F is the updated NetworkX Graph)
            print("Updated Nodes:", F.graph.nodes(data=True))

    return F


if __name__ == "__main__":
    import asyncio


    # Assuming all necessary imports and GraphDBType, get_graph_client, Document, DocumentType, etc. are defined

    # Initialize the graph client
    graph_client = get_graph_client(GraphDBType.NETWORKX)

    from typing import List, Type


    # Assuming generate_graph, KnowledgeGraph, and other necessary components are defined elsewhere

    async def generate_graphs_for_all_layers(text_input: str, layers: List[str], response_model: Type[BaseModel]):
        tasks = [generate_graph(text_input, "generate_graph_prompt.txt", {'layer': layer}, response_model) for layer in
                 layers]
        return await asyncio.gather(*tasks)


    input_article_one= "The quick brown fox jumps over the lazy dog"


    # Execute the async function and print results for each set of layers
    async def async_graph_per_layer(text_input: str, cognitive_layers: List[str]):
        graphs = await generate_graphs_for_all_layers(text_input, cognitive_layers, KnowledgeGraph)
        # for layer, graph in zip(cognitive_layers, graphs):
        #     print(f"{layer}: {graph}")
        return graphs

    cognitive_layers_one = ['Structural Layer', 'Semantic Layer',
         'Syntactic Layer',
         'Discourse Layer',
         'Pragmatic Layer',
         'Stylistic Layer',
         'Referential Layer',
         'Citation Layer',
         'Metadata Layer']
    required_layers_one = DefaultContentPrediction(
    label=TextContent(
        type='TEXT',
        subclass=[TextSubclass.ARTICLES]
        )
    )
    # Run the async function for each set of cognitive layers
    level_1_graph = asyncio.run( async_graph_per_layer(input_article_one, cognitive_layers_one))

    G = asyncio.run(append_to_graph(level_1_graph, required_layers_one, graph_client))



