""" Here we update semantic graph with content that classifier produced"""
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client, GraphDBType


async def add_classification_nodes(graph_client, document_id, classification_data):
    # graph_client = get_graph_client(GraphDBType.NETWORKX)
    #
    #
    # await graph_client.load_graph_from_file()

    data_type = classification_data["data_type"]
    layer_name = classification_data["layer_name"]

    # Create the layer classification node ID
    layer_classification_node_id = f"LLM_LAYER_CLASSIFICATION:{data_type}:{document_id}"

    # Add the node to the graph, unpacking the node data from the dictionary
    await graph_client.add_node(layer_classification_node_id, **classification_data)

    # Link this node to the corresponding document node
    await graph_client.add_edge(document_id, layer_classification_node_id, relationship = "classified_as")

    # Create the detailed classification node ID
    detailed_classification_node_id = f"LLM_CLASSIFICATION:LAYER:{layer_name}:{document_id}"

    # Add the detailed classification node, reusing the same node data
    await graph_client.add_node(detailed_classification_node_id, **classification_data)

    # Link the detailed classification node to the layer classification node
    await graph_client.add_edge(layer_classification_node_id, detailed_classification_node_id, relationship = "contains_analysis")

    return True
