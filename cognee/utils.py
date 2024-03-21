""" This module contains utility functions for the cognee. """

import os
import graphistry
from cognee.root_dir import get_absolute_path

def get_document_names(doc_input):
    """
    Get a list of document names.

    This function takes doc_input, which can be a folder path,
    a single document file path, or a document name as a string.
    It returns a list of document names based on the doc_input.

    Args:
        doc_input (str): The doc_input can be a folder path, a single document file path,
        or a document name as a string.

    Returns:
        list: A list of document names.

    Example usage:
        - Folder path: get_document_names(".data")
        - Single document file path: get_document_names(".data/example.pdf")
        - Document name provided as a string: get_document_names("example.docx")

    """
    if isinstance(doc_input, list):
        return doc_input
    if os.path.isdir(doc_input):
        # doc_input is a folder
        folder_path = doc_input
        document_names = []
        for filename in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, filename)):
                document_names.append(filename)
        return document_names
    elif os.path.isfile(doc_input):
        # doc_input is a single document file
        return [os.path.basename(doc_input)]
    elif isinstance(doc_input, str):
        # doc_input is a document name provided as a string
        return [doc_input]
    else:
        # doc_input is not valid
        return []


def format_dict(d):
    """Format a dictionary as a string."""
    # Initialize an empty list to store formatted items
    formatted_items = []

    # Iterate through all key-value pairs
    for key, value in d.items():
        # Format key-value pairs with a colon and space, and adding quotes for string values
        formatted_item = (
            f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}"
        )
        formatted_items.append(formatted_item)

    # Join all formatted items with a comma and a space
    formatted_string = ", ".join(formatted_items)

    # Add curly braces to mimic a dictionary
    formatted_string = f"{{{formatted_string}}}"

    return formatted_string


#
# How to render a graph
#
# import networkx as nx
#
# Create a simple NetworkX graph
# G = nx.Graph()
#
# # Add nodes
# G.add_node(1)
# G.add_node(2)
#
# Add an edge between nodes
# G.add_edge(1, 2)
#
# import asyncio
#
# Define the graph type (for this example, it's just a placeholder as the function doesn't use it yet)
# graph_type = "networkx"
#
# Call the render_graph function
# asyncio.run(render_graph(G, graph_type))
#
async def render_graph(graph, graph_type):
    # Authenticate with your Graphistry API key

    import networkx as nx
    from cognee.config import Config

    config = Config()
    config.load()

    graphistry.register(
        api = 3,
        username = config.graphistry_username,
        password = config.graphistry_password
    )

    # Convert the NetworkX graph to a Pandas DataFrame representing the edge list
    edges = nx.to_pandas_edgelist(graph)

    # Visualize the graph using Graphistry
    plotter = graphistry.edges(edges, "source", "target")

    # Visualize the graph (this will open a URL in your default web browser)
    url = plotter.plot(render = False, as_files = True)
    print(f"Graph is visualized at: {url}")
