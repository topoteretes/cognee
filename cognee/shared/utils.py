"""This module contains utility functions for the cognee."""

import os
from typing import BinaryIO, Union

import requests
import hashlib
from datetime import datetime, timezone
import graphistry
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
import nltk
import base64
import networkx as nx
from bokeh.io import output_file, save
from bokeh.plotting import figure, from_networkx
from bokeh.models import Circle, MultiLine, HoverTool, ColumnDataSource, Range1d


from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph import get_graph_engine

from uuid import uuid4
import pathlib

from cognee.shared.exceptions import IngestionError

# Analytics Proxy Url, currently hosted by Vercel
proxy_url = "https://test.prometh.ai"


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
            **additional_properties,
        },
    }

    response = requests.post(proxy_url, json=payload)

    if response.status_code != 200:
        print(f"Error sending telemetry through proxy: {response.status_code}")


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""

    # tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_file_content_hash(file_obj: Union[str, BinaryIO]) -> str:
    h = hashlib.md5()

    try:
        if isinstance(file_obj, str):
            with open(file_obj, "rb") as file:
                while True:
                    # Reading is buffered, so we can read smaller chunks.
                    chunk = file.read(h.block_size)
                    if not chunk:
                        break
                    h.update(chunk)
        else:
            while True:
                # Reading is buffered, so we can read smaller chunks.
                chunk = file_obj.read(h.block_size)
                if not chunk:
                    break
                h.update(chunk)

        return h.hexdigest()
    except IOError as e:
        raise IngestionError(message=f"Failed to load data from {file}: {e}")


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
    hex_colors = [
        "#%02x%02x%02x" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        for rgb in colors
    ]

    return dict(zip(unique_layers, hex_colors))


async def register_graphistry():
    config = get_base_config()
    graphistry.register(
        api=3, username=config.graphistry_username, password=config.graphistry_password
    )


def prepare_edges(graph, source, target, edge_key):
    edge_list = [
        {
            source: str(edge[0]),
            target: str(edge[1]),
            edge_key: str(edge[2]),
        }
        for edge in graph.edges(keys=True, data=True)
    ]

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
            node_size = (
                larger_size if any(keyword in str(node) for keyword in keywords) else default_size
            )
            node_data["size"] = node_size

        nodes_data.append(node_data)

    return pd.DataFrame(nodes_data)


async def render_graph(
    graph, include_nodes=False, include_color=False, include_size=False, include_labels=False
):
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
    plotter = plotter.bind(edge_label="relationship_name")

    if include_nodes:
        nodes = prepare_nodes(graph, include_size=include_size)
        plotter = plotter.nodes(nodes, "id")

        if include_size:
            plotter = plotter.bind(point_size="size")

        if include_color:
            pass
            # unique_layers = nodes["layer_description"].unique()
            # color_palette = generate_color_palette(unique_layers)
            # plotter = plotter.encode_point_color("layer_description", categorical_mapping=color_palette,
            #                                      default_mapping="silver")

        if include_labels:
            plotter = plotter.bind(point_label="name")

    # Visualization
    url = plotter.plot(render=False, as_files=True, memoize=False)
    print(f"Graph is visualized at: {url}")
    return url


# def sanitize_df(df):
#     """Replace NaNs and infinities in a DataFrame with None, making it JSON compliant."""
#     return df.replace([np.inf, -np.inf, np.nan], None)


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


import networkx as nx
from bokeh.plotting import figure, output_file, show
from bokeh.models import Circle, MultiLine, HoverTool, Range1d
from bokeh.io import output_notebook
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.plotting import figure, from_networkx
import base64
import cairosvg
import logging

logging.basicConfig(level=logging.INFO)


def convert_to_serializable_graph(G):
    """
    Convert a graph into a serializable format with stringified node and edge attributes.
    """

    (nodes, edges) = G
    networkx_graph = nx.MultiDiGraph()

    networkx_graph.add_nodes_from(nodes)
    networkx_graph.add_edges_from(edges)

    new_G = nx.MultiDiGraph() if isinstance(G, nx.MultiDiGraph) else nx.Graph()
    print(new_G)
    for node, data in new_G.nodes(data=True):
        serializable_data = {k: str(v) for k, v in data.items()}
        new_G.add_node(str(node), **serializable_data)
    for u, v, data in new_G.edges(data=True):
        serializable_data = {k: str(v) for k, v in data.items()}
        new_G.add_edge(str(u), str(v), **serializable_data)
    return new_G


def generate_layout_positions(G, layout_func, layout_scale):
    """
    Generate layout positions for the graph using the specified layout function.
    """
    positions = layout_func(G)
    return {str(node): (x * layout_scale, y * layout_scale) for node, (x, y) in positions.items()}


def assign_node_colors(G, node_attribute, palette):
    """
    Assign colors to nodes based on a specified attribute and a given palette.
    """
    unique_attrs = set(G.nodes[node].get(node_attribute, "Unknown") for node in G.nodes)
    color_map = {attr: palette[i % len(palette)] for i, attr in enumerate(unique_attrs)}
    return [color_map[G.nodes[node].get(node_attribute, "Unknown")] for node in G.nodes], color_map


def embed_logo(p, layout_scale, logo_alpha):
    """
    Embed a logo into the graph visualization as a watermark.
    """
    svg_logo = """<svg width="1294" height="324" viewBox="0 0 1294 324" fill="none" xmlns="http://www.w3.org/2000/svg">
        <mask id="mask0_103_2579" style="mask-type:alpha" maskUnits="userSpaceOnUse" x="0" y="0" width="1294" height="324">
        <path fill-rule="evenodd" clip-rule="evenodd" d="M380.648 131.09C365.133 131.09 353.428 142.843 353.428 156.285V170.258C353.428 183.7 365.133 195.452 380.648 195.452C388.268 195.452 393.57 193.212 401.288 187.611C405.57 184.506 411.579 185.449 414.682 189.714C417.805 193.978 416.842 199.953 412.561 203.038C402.938 209.995 393.727 214.515 380.628 214.515C355.49 214.555 334.241 195.197 334.241 170.258V156.285C334.241 131.366 355.49 112.008 380.648 112.008C393.747 112.008 402.958 116.528 412.581 123.485C416.862 126.59 417.805 132.545 414.702 136.809C411.579 141.074 405.589 142.017 401.308 138.912C393.59 133.331 388.268 131.071 380.667 131.071L380.648 131.09ZM474.875 131.09C459.792 131.09 447.557 143.255 447.557 158.289V168.509C447.557 183.543 459.792 195.708 474.875 195.708C489.958 195.708 501.977 183.602 501.977 168.509V158.289C501.977 143.196 489.879 131.09 474.875 131.09ZM428.37 158.289C428.37 132.741 449.188 112.008 474.875 112.008C500.563 112.008 521.164 132.8 521.164 158.289V168.509C521.164 193.998 500.622 214.79 474.875 214.79C449.129 214.79 428.37 194.057 428.37 168.509V158.289ZM584.774 131.601C569.652 131.601 557.457 143.747 557.457 158.683C557.457 173.618 569.672 185.764 584.774 185.764C599.877 185.764 611.876 173.697 611.876 158.683C611.876 143.668 599.818 131.601 584.774 131.601ZM538.269 158.683C538.269 133.154 559.126 112.519 584.774 112.519C595.693 112.519 605.67 116.253 613.545 122.483L620.733 115.329C624.484 111.595 630.552 111.595 634.303 115.329C638.054 119.063 638.054 125.096 634.303 128.83L625.819 137.281C629.178 143.688 631.063 150.979 631.063 158.702C631.063 184.152 610.501 204.866 584.774 204.866C584.519 204.866 584.264 204.866 584.008 204.866H563.643C560.226 204.866 557.457 207.617 557.457 211.017C557.457 214.417 560.226 217.168 563.643 217.168H589.939H598.345C605.258 217.168 612.426 219.075 618.18 223.614C624.131 228.292 627.901 235.229 628.569 243.739C629.747 258.812 619.123 269.11 610.482 272.431L586.444 283.004C581.593 285.127 575.937 282.945 573.796 278.131C571.655 273.316 573.855 267.675 578.686 265.553L602.96 254.882C603.137 254.803 603.333 254.724 603.51 254.665C604.531 254.292 606.259 253.191 607.614 251.364C608.871 249.674 609.598 247.649 609.421 245.252C609.146 241.754 607.811 239.808 606.259 238.609C604.551 237.253 601.84 236.271 598.325 236.271H564.036C563.937 236.271 563.839 236.271 563.721 236.271H563.604C549.601 236.271 538.23 224.97 538.23 211.037C538.23 201.997 543.002 194.077 550.171 189.616C542.747 181.44 538.23 170.612 538.23 158.722L538.269 158.683ZM694.045 131.601C679.021 131.601 666.825 143.727 666.825 158.683V205.239C666.825 210.506 662.525 214.79 657.242 214.79C651.959 214.79 647.658 210.526 647.658 205.239V158.683C647.658 133.193 668.436 112.519 694.065 112.519C719.693 112.519 740.471 133.193 740.471 158.683V205.239C740.471 210.506 736.17 214.79 730.887 214.79C725.605 214.79 721.304 210.526 721.304 205.239V158.683C721.304 143.727 709.128 131.601 694.084 131.601H694.045ZM807.204 131.621C791.748 131.621 779.356 143.963 779.356 159.017V168.843C779.356 183.897 791.748 196.238 807.204 196.238C812.565 196.238 817.514 194.745 821.698 192.19C826.214 189.439 832.126 190.834 834.895 195.334C837.664 199.835 836.25 205.711 831.733 208.462C824.604 212.825 816.179 215.321 807.204 215.321C781.3 215.321 760.169 194.588 760.169 168.843V159.017C760.169 133.272 781.3 112.538 807.204 112.538C829.357 112.538 847.778 127.671 852.707 148.07L854.632 156.049L813.744 172.597C808.834 174.581 803.237 172.243 801.234 167.349C799.231 162.475 801.587 156.894 806.497 154.909L830.947 145.004C826.156 136.986 817.338 131.601 807.165 131.601L807.204 131.621ZM912.37 131.621C896.914 131.621 884.522 143.963 884.522 159.017V168.843C884.522 183.897 896.914 196.238 912.37 196.238C917.732 196.238 922.681 194.745 926.864 192.19C928.965 190.913 930.89 189.36 932.559 187.572C936.192 183.72 942.261 183.543 946.11 187.139C949.979 190.736 950.175 196.789 946.542 200.621C943.694 203.628 940.454 206.281 936.879 208.462C929.731 212.825 921.326 215.321 912.331 215.321C886.427 215.321 865.296 194.588 865.296 168.843V159.017C865.296 133.272 886.427 112.538 912.331 112.538C934.484 112.538 952.905 127.671 957.834 148.07L959.759 156.049L918.871 172.597C913.961 174.581 908.364 172.243 906.361 167.349C904.358 162.475 906.714 156.894 911.624 154.909L936.074 145.004C931.282 136.986 922.465 131.601 912.292 131.601L912.37 131.621Z" fill="#6510F4"/>
        </mask>
        <g mask="url(#mask0_103_2579)">
        <rect x="86" y="-119" width="1120" height="561" fill="#6510F4"/>
        </g>
        </svg>"""  # Add your SVG content here
    png_data = cairosvg.svg2png(bytestring=svg_logo.encode("utf-8"))
    logo_base64 = base64.b64encode(png_data).decode("utf-8")
    logo_url = f"data:image/png;base64,{logo_base64}"
    p.image_url(
        url=[logo_url],
        x=-layout_scale * 0.5,
        y=layout_scale * 0.5,
        w=layout_scale,
        h=layout_scale,
        anchor="center",
        global_alpha=logo_alpha,
    )


def style_and_render_graph(p, G, layout_positions, node_attribute, node_colors, centrality):
    """
    Apply styling and render the graph into the plot.
    """
    graph_renderer = from_networkx(G, layout_positions)
    node_radii = [0.02 + 0.1 * centrality[node] for node in G.nodes()]
    graph_renderer.node_renderer.data_source.data["radius"] = node_radii
    graph_renderer.node_renderer.data_source.data["fill_color"] = node_colors
    graph_renderer.node_renderer.glyph = Circle(
        radius="radius",
        fill_color="fill_color",
        fill_alpha=0.9,
        line_color="#000000",
        line_width=1.5,
    )
    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="#000000",
        line_alpha=0.3,
        line_width=1.5,
    )
    p.renderers.append(graph_renderer)
    return graph_renderer


def create_cognee_style_network_with_logo(
    G,
    output_filename="cognee_network_with_logo.html",
    title="Cognee-Style Network",
    node_attribute="group",
    layout_func=nx.spring_layout,
    layout_scale=3.0,
    logo_alpha=0.1,
):
    """
    Create a Cognee-inspired network visualization with an embedded logo.
    """
    logging.info("Converting graph to serializable format...")
    G = convert_to_serializable_graph(G)

    logging.info("Generating layout positions...")
    layout_positions = generate_layout_positions(G, layout_func, layout_scale)

    logging.info("Assigning node colors...")
    palette = ["#6510F4", "#0DFF00", "#FFFFFF"]
    node_colors, color_map = assign_node_colors(G, node_attribute, palette)

    logging.info("Calculating centrality...")
    centrality = nx.degree_centrality(G)

    logging.info("Preparing Bokeh output...")
    output_file(output_filename)
    p = figure(
        title=title,
        tools="pan,wheel_zoom,save,reset,hover",
        active_scroll="wheel_zoom",
        width=1200,
        height=900,
        background_fill_color="#F4F4F4",
        x_range=Range1d(-layout_scale, layout_scale),
        y_range=Range1d(-layout_scale, layout_scale),
    )
    p.toolbar.logo = None
    p.axis.visible = False
    p.grid.visible = False

    logging.info("Embedding logo into visualization...")
    embed_logo(p, layout_scale, logo_alpha)

    logging.info("Styling and rendering graph...")
    style_and_render_graph(p, G, layout_positions, node_attribute, node_colors, centrality)

    logging.info("Adding hover tool...")
    hover_tool = HoverTool(
        tooltips=[
            ("Node", "@index"),
            (node_attribute.capitalize(), f"@{node_attribute}"),
            ("Centrality", "@radius{0.00}"),
        ],
    )
    p.add_tools(hover_tool)

    logging.info(f"Saving visualization to {output_filename}...")
    html_content = file_html(p, CDN, title)
    with open(output_filename, "w") as f:
        f.write(html_content)

    logging.info("Visualization complete.")
    return html_content


def graph_to_tuple(graph):
    """
    Converts a networkx graph to a tuple of (nodes, edges).

    :param graph: A networkx graph.
    :return: A tuple (nodes, edges).
    """
    nodes = list(graph.nodes(data=True))  # Get nodes with attributes
    edges = list(graph.edges(data=True))  # Get edges with attributes
    return (nodes, edges)


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    import networkx as nx

    # Create a sample graph
    nodes = [
        (1, {"group": "A"}),
        (2, {"group": "A"}),
        (3, {"group": "B"}),
        (4, {"group": "B"}),
        (5, {"group": "C"}),
    ]
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]

    # Create a NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Call the function
    output_html = create_cognee_style_network_with_logo(
        G=G,
        output_filename="example_network.html",
        title="Example Cognee Network",
        node_attribute="group",  # Attribute to use for coloring nodes
        layout_func=nx.spring_layout,  # Layout function
        layout_scale=3.0,  # Scale for the layout
        logo_alpha=0.2,  # Transparency of the logo
    )

    # Print the output filename
    print("Network visualization saved as example_network.html")

#     # Create a random geometric graph
#     G = nx.random_geometric_graph(50, 0.3)
#     # Assign random group attributes for coloring
#     for i, node in enumerate(G.nodes()):
#         G.nodes[node]["group"] = f"Group {i % 3 + 1}"
#
#     create_cognee_graph(
#         G,
#         output_filename="cognee_style_network_with_logo.html",
#         title="Cognee-Graph Network",
#         node_attribute="group",
#         layout_func=nx.spring_layout,
#         layout_scale=3.0,  # Replace with your logo file path
#     )
