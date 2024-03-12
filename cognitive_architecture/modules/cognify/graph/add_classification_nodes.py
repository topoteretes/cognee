""" Here we update semantic graph with content that classifier produced"""
from cognitive_architecture.infrastructure.databases.graph.get_graph_client import get_graph_client, GraphDBType


async def add_classification_nodes(graph_id, classification_data):
    graph_client = get_graph_client(GraphDBType.NETWORKX)

    await graph_client.load_graph_from_file()

    context = classification_data["context_name"]
    layer = classification_data["layer_name"]

    # Create the layer classification node ID using the context_name
    layer_classification_node_id = f"LLM_LAYER_CLASSIFICATION:{context}:{graph_id}"

    # Add the node to the graph, unpacking the node data from the dictionary
    await graph_client.add_node(layer_classification_node_id, **classification_data)

    # Link this node to the corresponding document node
    await graph_client.add_edge(graph_id, layer_classification_node_id, relationship = "classified_as")

    # Create the detailed classification node ID using the context_name
    detailed_classification_node_id = f"LLM_CLASSIFICATION:LAYER:{layer}:{graph_id}"

    # Add the detailed classification node, reusing the same node data
    await graph_client.add_node(detailed_classification_node_id, **classification_data)

    # Link the detailed classification node to the layer classification node
    await graph_client.add_edge(layer_classification_node_id, detailed_classification_node_id, relationship = "contains_analysis")

    return True



# if __name__ == "__main__":
#     import asyncio

#     # Assuming all necessary imports and GraphDBType, get_graph_client, Document, DocumentType, etc. are defined

#     # Initialize the graph client
#     graph_client = get_graph_client(GraphDBType.NETWORKX)


#     G = asyncio.run(add_classification_nodes(graph_client, "Document:doc1", {"data_type": "text",
#  "context_name": "TEXT",
#  "layer_name": "Articles, essays, and reports"}))

#     from cognitive_architecture.utils import render_graph
#     ff = asyncio.run( render_graph(G.graph, graph_type='networkx'))
#     print(ff)