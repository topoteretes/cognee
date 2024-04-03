""" This module contains utility functions for the cognee. """

import os
import graphistry
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer


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
    # config = Config()
    # config.load()
    #
    # # Register with Graphistry using API key
    # graphistry.register(api=3, username=config.graphistry_username, password=config.graphistry_password)
    #
    # # Convert your NetworkX graph edges and nodes to Pandas DataFrame
    # edges = nx.to_pandas_edgelist(graph)
    #
    #
    # # Prepare nodes DataFrame with "id" and "layer_description"
    # nodes_data = [{"id": node, "layer_description": graph.nodes[node]["layer_description"]}
    #               for node in graph.nodes if "layer_description" in graph.nodes[node]]
    # nodes = pd.DataFrame(nodes_data)
    #
    # # Visualize the graph using Graphistry
    # plotter = graphistry.edges(edges, "source", "target").nodes(nodes, "id")
    #
    # # Generate a dynamic color palette based on unique "layer_description" values
    # if nodes["layer_description"]:
    #     unique_layers = nodes["layer_description"].unique()
    #     color_palette = generate_color_palette(unique_layers)
    #
    #     plotter = plotter.encode_point_color(
    #         "layer_description",
    #         categorical_mapping = color_palette,
    #         default_mapping = "silver"  # Default color if any "layer_description" is not in the mapping
    #     )
    #
    # # Visualize the graph (this will open a URL in your default web browser)
    # url = plotter.plot(render = False, as_files = True)
    # print(f"Graph is visualized at: {url}")


def generate_color_palette(unique_layers):
    colormap = plt.cm.get_cmap("viridis", len(unique_layers))
    colors = [colormap(i) for i in range(len(unique_layers))]
    hex_colors = ["#%02x%02x%02x" % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)) for rgb in colors]

    return dict(zip(unique_layers, hex_colors))

async def render_graph(graph):
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
    # nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    #
    # df['named_entities'] = df['named_entities'].apply(lambda x: x if isinstance(x, list) else [x])

    # Visualize the graph using Graphistry
    plotter = graphistry.edges(edges, "source", "target")
    # .nodes(nodes, 'index'))

    # Bind the 'name' column as the node label
    # plotter = plotter.bind(node='index', point_title='name')

    # Visualize the graph (this will open a URL in your default web browser)
    url = plotter.plot(render = False, as_files = True)
    print(f"Graph is visualized at: {url}")

# async def render_graph(graph):
#     # Authenticate with your Graphistry API key
#
#     import networkx as nx
#     from cognee.config import Config
#
#     config = Config()
#     config.load()
#
#     graphistry.register(
#         api=3,
#         username=config.graphistry_username,
#         password=config.graphistry_password
#     )
#
#     # Convert the NetworkX graph to a Pandas DataFrame representing the edge list
#     edges = nx.to_pandas_edgelist(graph)
#
#     # Prepare nodes DataFrame with "id" and "layer_description"
#     nodes_data = [{"id": node, "layer_description": graph.nodes[node]["layer_description"]}
#                   for node in graph.nodes if "layer_description" in graph.nodes[node]]
#     nodes = pd.DataFrame(nodes_data)
#
#     # Visualize the graph using Graphistry
#     plotter = graphistry.edges(edges, "source", "target").nodes(nodes, "id")
#
#     # Generate a dynamic color palette based on unique "layer_description" values
#     if 'layer_description' in nodes:
#         unique_layers = nodes["layer_description"].unique()
#         color_palette = generate_color_palette(unique_layers)
#
#         plotter = plotter.encode_point_color(
#             "layer_description",
#             categorical_mapping=color_palette,
#             default_mapping="silver"  # Default color if any "layer_description" is not in the mapping
#         )
#
#     # Visualize the graph (this will open a URL in your default web browser)
#     url = plotter.plot(render=False, as_files=True)
#     print(f"Graph is visualized at: {url}")
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Ensure that the necessary NLTK resources are downloaded
nltk.download('maxent_ne_chunker')
nltk.download('words')

# The sentence you want to tag and recognize entities in
sentence = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California."


async def extract_pos_tags(sentence):
    """Extract Part-of-Speech (POS) tags for words in a sentence."""
    # Tokenize the sentence into words
    tokens = word_tokenize(sentence)

    # Tag each word with its corresponding POS tag
    pos_tags = pos_tag(tokens)

    return pos_tags


async def extract_named_entities(sentence):
    """Extract Named Entities from a sentence."""
    # Tokenize the sentence into words
    tokens = word_tokenize(sentence)

    # Perform POS tagging on the tokenized sentence
    tagged = pos_tag(tokens)

    # Perform Named Entity Recognition (NER) on the tagged tokens
    entities = ne_chunk(tagged)

    return entities

nltk.download('vader_lexicon')

async def extract_sentiment_vader(text):
    """
    Analyzes the sentiment of a given text using the VADER Sentiment Intensity Analyzer.

    Parameters:
    text (str): The text to analyze.

    Returns:
    dict: A dictionary containing the polarity scores for the text.
    """
    # Initialize the VADER Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    # Obtain the polarity scores for the text
    polarity_scores = sia.polarity_scores(text)

    return polarity_scores


if __name__ == "__main__":
    sample_text = "I love sunny days, but I hate the rain."
    sentiment_scores = extract_sentiment_vader(sample_text)
    print("Sentiment analysis results:", sentiment_scores)
