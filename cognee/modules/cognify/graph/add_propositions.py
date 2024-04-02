""" Here we update semantic graph with content that classifier produced"""
import uuid
import json
from datetime import datetime

from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client, GraphDBType
from cognee.shared.encode_uuid import encode_uuid


async def add_propositions(graph_client,
    data_type,
    layer_name,
    layer_description,
    new_data,
    layer_uuid,
    layer_decomposition_uuid
):
    """ Add nodes and edges to the graph for the given LLM knowledge graph and the layer"""
    # graph_client = get_graph_client(GraphDBType.NETWORKX)
    #
    # await graph_client.load_graph_from_file()

    layer_node_id = None
    print(f"Looking for layer '{layer_name}' under category '{data_type}'")
    if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:
        for node_id, data in graph_client.graph.nodes(data = True):
            if layer_name in node_id:
                layer_node_id = node_id
    elif infrastructure_config.get_config()["graph_engine"] == GraphDBType.NEO4J:
        layer_node_id = await graph_client.filter_nodes(search_node='node_id', search_criteria=layer_name)['d']['node_id']

    if not layer_node_id:
        print(f"Subclass '{layer_name}' under category '{data_type}' not found in the graph.")
        return graph_client

    # Mapping from old node IDs to new node IDs
    node_id_mapping = {}

    # Add nodes from the Pydantic object
    for node in new_data.nodes:
        unique_node_id = uuid.uuid4()

        new_node_id = f"{node.description} - {str(layer_uuid)}  - {str(layer_decomposition_uuid)} - {str(unique_node_id)}"

        await graph_client.add_node(
            new_node_id,
            description=node.description,
            layer_uuid=str(layer_uuid),
            layer_description=str(layer_description),
            layer_decomposition_uuid=str(layer_decomposition_uuid),
            unique_id=str(unique_node_id),
            type='detail'
        )

        print("HERE IS LAYER NODE ID", layer_node_id)

        print("HERE IS NEW NODE ID", new_node_id)

        await graph_client.add_edge(layer_node_id, new_node_id, relationship_type='detail')

        # Store the mapping from old node ID to new node ID
        node_id_mapping[node.id] = new_node_id

    # Add edges from the Pydantic object using the new node IDs
    for edge in new_data.edges:
        # Use the mapping to get the new node IDs
        source_node_id = node_id_mapping.get(edge.source)
        target_node_id = node_id_mapping.get(edge.target)

        if source_node_id and target_node_id:
            await graph_client.add_edge(source_node_id, target_node_id, description=edge.description, relationship_type=edge.description)
        else:
            print(f"Could not find mapping for edge from {edge.source} to {edge.target}")

async def append_to_graph(graph_client, layer_graphs, required_layers):
    # Generate a UUID for the overall layer
    layer_uuid = uuid.uuid4()
    print("EXAMPLE OF LAYER GRAPHS", required_layers)
    # Extract category name from required_layers data
    data_type = required_layers["data_type"]

    # Extract subgroup name from required_layers data
    # Assuming there's always at least one layer and we're taking the first
    layer_name = required_layers["layer_name"]

    for layer_ind in layer_graphs:

        for layer_json, knowledge_graph in layer_ind.items():
            # Decode the JSON key to get the layer description
            layer_description = json.loads(layer_json)

            # Generate a UUID for this particular layer decomposition
            layer_decomposition_id = encode_uuid(uuid.uuid4())

            # Assuming append_data_to_graph is defined elsewhere and appends data to graph_client
            # You would pass relevant information from knowledge_graph along with other details to this function
            print("ADDING PROPOSITIONS")
            await add_propositions(
                graph_client,
                data_type,
                layer_name,
                layer_description,
                knowledge_graph,
                layer_uuid,
                layer_decomposition_id
            )


# async def bubu():
#
#     graph_client = await get_graph_client(GraphDBType.NEO4J)
#
#     from typing import List, Type
#
#
#     # Assuming generate_graph, KnowledgeGraph, and other necessary components are defined elsewhere
#
#     from cognee.shared.data_models import KnowledgeGraph, DefaultCognitiveLayer, TextContent, TextSubclass, DefaultContentPrediction
#     from cognee.modules.cognify.llm.generate_graph import generate_graph
#     async def generate_graphs_for_all_layers(text_input: str, layers: List[str], response_model= KnowledgeGraph):
#         tasks = [generate_graph(text_input, "generate_graph_prompt.txt", {'layer': layer}, response_model) for layer in
#                  layers]
#         return await asyncio.gather(*tasks)
#
#
#     input_article_one= "The quick brown fox jumps over the lazy dog"
#
#
#     # Execute the async function and print results for each set of layers
#     async def async_graph_per_layer(text_input: str, cognitive_layers: List[str]):
#         graphs = await generate_graphs_for_all_layers(text_input, cognitive_layers, KnowledgeGraph)
#         # for layer, graph in zip(cognitive_layers, graphs):
#         #     print(f"{layer}: {graph}")
#         return graphs
#
#     cognitive_layers_one = ['Structural Layer', 'Semantic Layer',
#          'Syntactic Layer',
#          'Discourse Layer',
#          'Pragmatic Layer',
#          'Stylistic Layer',
#          'Referential Layer',
#          'Citation Layer',
#          'Metadata Layer']
#     required_layers_one = DefaultContentPrediction(
#     label=TextContent(
#         type='TEXT',
#         subclass=[TextSubclass.ARTICLES]
#         )
#     )
#     # Run the async function for each set of cognitive layers
#     level_1_graph = asyncio.run( async_graph_per_layer(input_article_one, cognitive_layers_one))
#
#     G = await append_to_graph(level_1_graph, required_layers_one, graph_client)
#
# if __name__ == "__main__":
#
#     import asyncio
#
#
#     # Assuming all necessary imports and GraphDBType, get_graph_client, Document, DocumentType, etc. are defined
#
#     # Initialize the graph client
#
#
#
#     G = asyncio.run(bubu())


import asyncio
from typing import List
from cognee.shared.data_models import KnowledgeGraph, DefaultCognitiveLayer, TextContent, TextSubclass, DefaultContentPrediction
from cognee.modules.cognify.llm.generate_graph import generate_graph
# Assuming get_graph_client, generate_graph, KnowledgeGraph, and other necessary components are defined elsewhere

async def generate_graphs_for_all_layers(text_input: str, layers: List[str], response_model):
    tasks = [generate_graph(text_input, "generate_graph_prompt.txt", {'layer': layer}, response_model) for layer in layers]
    return await asyncio.gather(*tasks)

async def async_graph_per_layer(text_input: str, cognitive_layers: List[str], response_model):
    graphs = await generate_graphs_for_all_layers(text_input, cognitive_layers, response_model)
    return graphs

# async def append_to_graph(graph_data, prediction_layers, client):
#     # Your logic to append data to the graph goes here
#     pass

async def bubu():
    infrastructure_config.set_config({
        "graph_engine": GraphDBType.NEO4J
    })
    graph_client = await get_graph_client(GraphDBType.NEO4J)

    input_article_one = "The quick brown fox jumps over the lazy dog"
    cognitive_layers_one = [
        'Structural Layer', 'Semantic Layer', 'Syntactic Layer',
        'Discourse Layer', 'Pragmatic Layer', 'Stylistic Layer',
        'Referential Layer', 'Citation Layer', 'Metadata Layer'
    ]
    required_layers_one = DefaultContentPrediction(
        label=TextContent(
            type='TEXT',
            subclass=[TextSubclass.ARTICLES]
        )
    )
    print("Running async_graph_per_layer", required_layers_one)
    categories = [subclass_attribute.name for subclass_attribute in required_layers_one.label.subclass]


    # Directly await the coroutine for generating and appending to graph
    level_1_graph = await async_graph_per_layer(input_article_one, cognitive_layers_one, KnowledgeGraph)

    print("GRAPH is ", level_1_graph)
    await append_to_graph(graph_client= graph_client, layer_graphs=level_1_graph, required_layers={'data_type': 'text', 'layer_name': 'Research papers and academic publications'})

    # Return something if needed, for now just return None
    return None

# This is the correct place to use asyncio.run() - at the top level of the program
if __name__ == "__main__":
    asyncio.run(bubu())