""" This module contains utility functions for the cognee. """

import os
import uuid
import datetime
import graphistry
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
import nltk
from posthog import Posthog

from cognee.config import Config

config = Config()
config.load()


def send_telemetry(event_name: str):
    if os.getenv("TELEMETRY_DISABLED"):
        return

    env = os.getenv("ENV")
    if env in ["local", "test", "dev"]:
        return

    posthog = Posthog(
        project_api_key = "phc_bbR86N876kwub62Lr3dhQ7zIeRyMMMm0fxXqxPqzLm3",
        host="https://eu.i.posthog.com"
    )

    user_id = str(uuid.uuid4())
    current_time = datetime.datetime.now()
    properties = {
        "time": current_time.strftime("%m/%d/%Y")
    }

    try:
        posthog.capture(user_id, event_name, properties)
    except Exception as e:
        print("ERROR sending telemetric data to Posthog. See exception: %s", e)

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

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""

    # tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def trim_text_to_max_tokens(text: str, max_tokens: int, encoding_name: str) -> str:
    """
    Trims the text so that the number of tokens does not exceed max_tokens.

    Args:
    text (str): Original text string to be trimmed.
    max_tokens (int): Maximum number of tokens allowed.
    encoding_name (str): The name of the token encoding to use.

    Returns:
    str: Trimmed version of text or original text if under the limit.
    """
    # First check the number of tokens
    num_tokens = num_tokens_from_string(text, encoding_name)

    # If the number of tokens is within the limit, return the text as is
    if num_tokens <= max_tokens:
        return text

    # If the number exceeds the limit, trim the text
    # This is a simple trim, it may cut words in half; consider using word boundaries for a cleaner cut
    encoded_text = tiktoken.get_encoding(encoding_name).encode(text)
    trimmed_encoded_text = encoded_text[:max_tokens]
    # Decoding the trimmed text
    trimmed_text = tiktoken.get_encoding(encoding_name).decode(trimmed_encoded_text)
    return trimmed_text


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


def generate_color_palette(unique_layers):
    colormap = plt.cm.get_cmap("viridis", len(unique_layers))
    colors = [colormap(i) for i in range(len(unique_layers))]
    hex_colors = ["#%02x%02x%02x" % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)) for rgb in colors]

    return dict(zip(unique_layers, hex_colors))


async def register_graphistry():
    graphistry.register(api = 3, username = config.graphistry_username, password = config.graphistry_password)


def prepare_edges(graph):
    return nx.to_pandas_edgelist(graph)


def prepare_nodes(graph, include_size=False):
    nodes_data = []
    for node in graph.nodes:
        node_info = graph.nodes[node]
        description = node_info.get("layer_description", {}).get("layer", "Default Layer") if isinstance(
            node_info.get("layer_description"), dict) else node_info.get("layer_description", "Default Layer")
        # description = node_info['layer_description']['layer'] if isinstance(node_info.get('layer_description'), dict) and 'layer' in node_info['layer_description'] else node_info.get('layer_description', node)
        # if isinstance(node_info.get('layer_description'), dict) and 'layer' in node_info.get('layer_description'):
        #     description = node_info['layer_description']['layer']
        # # Use 'layer_description' directly if it's not a dictionary, otherwise default to node ID
        # else:
        #     description = node_info.get('layer_description', node)

        node_data = {"id": node, "layer_description": description}
        if include_size:
            default_size = 10  # Default node size
            larger_size = 20  # Size for nodes with specific keywords in their ID
            keywords = ["DOCUMENT", "User", "LAYER"]
            node_size = larger_size if any(keyword in str(node) for keyword in keywords) else default_size
            node_data["size"] = node_size
        nodes_data.append(node_data)

    return pd.DataFrame(nodes_data)


async def render_graph(graph, include_nodes=False, include_color=False, include_size=False, include_labels=False):
    await register_graphistry()
    edges = prepare_edges(graph)
    plotter = graphistry.edges(edges, "source", "target")

    if include_nodes:
        nodes = prepare_nodes(graph, include_size=include_size)
        plotter = plotter.nodes(nodes, "id")


        if include_size:
            plotter = plotter.bind(point_size="size")


        if include_color:
            unique_layers = nodes["layer_description"].unique()
            color_palette = generate_color_palette(unique_layers)
            plotter = plotter.encode_point_color("layer_description", categorical_mapping=color_palette,
                                                 default_mapping="silver")


        if include_labels:
            plotter = plotter.bind(point_label = "layer_description")



    # Visualization
    url = plotter.plot(render=False, as_files=True, memoize=False)
    print(f"Graph is visualized at: {url}")


def sanitize_df(df):
    """Replace NaNs and infinities in a DataFrame with None, making it JSON compliant."""
    return df.replace([np.inf, -np.inf, np.nan], None)


def get_entities(tagged_tokens):
    nltk.download("maxent_ne_chunker", quiet=True)
    from nltk.chunk import ne_chunk
    return ne_chunk(tagged_tokens)


def extract_pos_tags(sentence):
    """Extract Part-of-Speech (POS) tags for words in a sentence."""

    # Ensure that the necessary NLTK resources are downloaded
    nltk.download("words", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize

    # Tokenize the sentence into words
    tokens = word_tokenize(sentence)

    # Tag each word with its corresponding POS tag
    pos_tags = pos_tag(tokens)

    return pos_tags


def extract_named_entities(sentence):
    """Extract Named Entities from a sentence."""
    # Tokenize the sentence into words
    tagged_tokens = extract_pos_tags(sentence)

    # Perform Named Entity Recognition (NER) on the tagged tokens
    entities = get_entities(tagged_tokens)

    return entities


def extract_sentiment_vader(text):
    """
    Analyzes the sentiment of a given text using the VADER Sentiment Intensity Analyzer.

    Parameters:
    text (str): The text to analyze.

    Returns:
    dict: A dictionary containing the polarity scores for the text.
    """
    from nltk.sentiment import SentimentIntensityAnalyzer

    nltk.download("vader_lexicon", quiet=True)

    # Initialize the VADER Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    # Obtain the polarity scores for the text
    polarity_scores = sia.polarity_scores(text)

    return polarity_scores


if __name__ == "__main__":
    sample_text = "I love sunny days, but I hate the rain."
    sentiment_scores = extract_sentiment_vader(sample_text)
    print("Sentiment analysis results:", sentiment_scores)
