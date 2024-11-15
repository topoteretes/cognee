""" This module contains utility functions for the cognee. """
import os
import requests
from datetime import datetime, timezone
import graphistry
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
import nltk

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph import get_graph_engine

from uuid import uuid4
import pathlib

# Analytics Proxy Url, currently hosted by Vercel
vercel_url = "https://proxyanalytics.vercel.app"

def get_anonymous_id():
    """Creates or reads a anonymous user id"""
    home_dir = str(pathlib.Path(pathlib.Path(__file__).parent.parent.parent.resolve()))

    if not os.path.isdir(home_dir):
        os.makedirs(home_dir, exist_ok=True)
    anonymous_id_file = os.path.join(home_dir, ".anon_id")
    if not os.path.isfile(anonymous_id_file):
        anonymous_id = str(uuid4())
        with open(anonymous_id_file, "w", encoding="utf-8") as f:
            f.write(anonymous_id)
    else:
        with open(anonymous_id_file, "r", encoding="utf-8") as f:
            anonymous_id = f.read()
    return anonymous_id

def send_telemetry(event_name: str, user_id, additional_properties: dict = {}):
    if os.getenv("TELEMETRY_DISABLED"):
        return

    env = os.getenv("ENV")
    if env in ["test", "dev"]:
        return

    current_time = datetime.now(timezone.utc)
    payload = {
        "anonymous_id": str(get_anonymous_id()),
        "event_name": event_name,
        "user_properties": {
            "user_id": str(user_id),
        },
        "properties": {
            "time": current_time.strftime("%m/%d/%Y"),
            "user_id": str(user_id),
            **additional_properties
        },
    }

    response = requests.post(vercel_url, json=payload)

    if response.status_code != 200:
        print(f"Error sending telemetry through proxy: {response.status_code}")

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


def generate_color_palette(unique_layers):
    colormap = plt.cm.get_cmap("viridis", len(unique_layers))
    colors = [colormap(i) for i in range(len(unique_layers))]
    hex_colors = ["#%02x%02x%02x" % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)) for rgb in colors]

    return dict(zip(unique_layers, hex_colors))


async def register_graphistry():
    config = get_base_config()
    graphistry.register(api = 3, username = config.graphistry_username, password = config.graphistry_password)


def prepare_edges(graph, source, target, edge_key):
    edge_list = [{
        source: str(edge[0]),
        target: str(edge[1]),
        edge_key: str(edge[2]),
    } for edge in graph.edges(keys = True, data = True)]

    return pd.DataFrame(edge_list)


def prepare_nodes(graph, include_size=False):
    nodes_data = []
    for node in graph.nodes:
        node_info = graph.nodes[node]

        if not node_info:
            continue

        node_data = {
            "id": str(node),
            "name": node_info["name"] if "name" in node_info else str(node),
        }

        if include_size:
            default_size = 10  # Default node size
            larger_size = 20  # Size for nodes with specific keywords in their ID
            keywords = ["DOCUMENT", "User"]
            node_size = larger_size if any(keyword in str(node) for keyword in keywords) else default_size
            node_data["size"] = node_size

        nodes_data.append(node_data)

    return pd.DataFrame(nodes_data)


async def render_graph(graph, include_nodes=False, include_color=False, include_size=False, include_labels=False):
    await register_graphistry()

    if not isinstance(graph, nx.MultiDiGraph):
        graph_engine = await get_graph_engine()
        networkx_graph = nx.MultiDiGraph()

        (nodes, edges) = await graph_engine.get_graph_data()

        networkx_graph.add_nodes_from(nodes)
        networkx_graph.add_edges_from(edges)

        graph = networkx_graph

    edges = prepare_edges(graph, "source_node", "target_node", "relationship_name")
    plotter = graphistry.edges(edges, "source_node", "target_node")
    plotter = plotter.bind(edge_label = "relationship_name")

    if include_nodes:
        nodes = prepare_nodes(graph, include_size = include_size)
        plotter = plotter.nodes(nodes, "id")

        if include_size:
            plotter = plotter.bind(point_size = "size")


        if include_color:
            pass
            # unique_layers = nodes["layer_description"].unique()
            # color_palette = generate_color_palette(unique_layers)
            # plotter = plotter.encode_point_color("layer_description", categorical_mapping=color_palette,
            #                                      default_mapping="silver")


        if include_labels:
            plotter = plotter.bind(point_label = "name")


    # Visualization
    url = plotter.plot(render=False, as_files=True, memoize=False)
    print(f"Graph is visualized at: {url}")
    return url


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
