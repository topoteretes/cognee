""" Here we update semantic graph with content that classifier produced"""
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client, GraphDBType


async def add_summary_nodes(graph_client,document_id, classification_data):
    # graph_client = get_graph_client(GraphDBType.NETWORKX)
    #
    # await graph_client.load_graph_from_file()


    # Create the layer classification node ID
    layer_classification_node_id = f"LLM_LAYER_SUMMARY_{document_id}"

    # Add the node to the graph, unpacking the node data from the dictionary
    await graph_client.add_node(layer_classification_node_id, **classification_data)

    # Link this node to the corresponding document node
    await graph_client.add_edge(document_id, layer_classification_node_id, relationship_type = "summarized_as")

    # Create the detailed classification node ID
    detailed_classification_node_id = f"LLM_SUMMARY_LAYER_{document_id}"

    # Add the detailed classification node, reusing the same node data
    await graph_client.add_node(detailed_classification_node_id, **classification_data)

    # Link the detailed classification node to the layer classification node
    await graph_client.add_edge(layer_classification_node_id, detailed_classification_node_id, relationship_type = "contains_summary")

    return True
