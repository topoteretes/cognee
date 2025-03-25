"""This module contains utility functions for the cognee."""

import os
from typing import BinaryIO, Union

import requests
import hashlib
from datetime import datetime, timezone
import graphistry
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import http.server
import socketserver
from threading import Thread
import sys

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph import get_graph_engine

from uuid import uuid4
import pathlib
from cognee.shared.exceptions import IngestionError

# Analytics Proxy Url, currently hosted by Vercel
proxy_url = "https://test.prometh.ai"


def get_entities(tagged_tokens):
    import nltk

    nltk.download("maxent_ne_chunker", quiet=True)

    from nltk.chunk import ne_chunk

    return ne_chunk(tagged_tokens)


def extract_pos_tags(sentence):
    """Extract Part-of-Speech (POS) tags for words in a sentence."""
    import nltk

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
            **node_info,
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
    graph=None, include_nodes=True, include_color=False, include_size=False, include_labels=True
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


async def convert_to_serializable_graph(G):
    """
    Convert a graph into a serializable format with stringified node and edge attributes.
    """

    (nodes, edges) = G

    networkx_graph = nx.MultiDiGraph()
    networkx_graph.add_nodes_from(nodes)
    networkx_graph.add_edges_from(edges)

    # Create a new graph to store the serializable version
    new_G = nx.MultiDiGraph()

    # Serialize nodes
    for node, data in networkx_graph.nodes(data=True):
        serializable_data = {k: str(v) for k, v in data.items()}
        new_G.add_node(str(node), **serializable_data)

    # Serialize edges
    for u, v, data in networkx_graph.edges(data=True):
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


def embed_logo(p, layout_scale, logo_alpha, position):
    """
    Embed a logo into the graph visualization as a watermark.
    """

    # svg_logo = """<svg width="1294" height="324" viewBox="0 0 1294 324" fill="none" xmlns="http://www.w3.org/2000/svg">
    #     <mask id="mask0_103_2579" style="mask-type:alpha" maskUnits="userSpaceOnUse" x="0" y="0" width="1294" height="324">
    #     <path fill-rule="evenodd" clip-rule="evenodd" d="M380.648 131.09C365.133 131.09 353.428 142.843 353.428 156.285V170.258C353.428 183.7 365.133 195.452 380.648 195.452C388.268 195.452 393.57 193.212 401.288 187.611C405.57 184.506 411.579 185.449 414.682 189.714C417.805 193.978 416.842 199.953 412.561 203.038C402.938 209.995 393.727 214.515 380.628 214.515C355.49 214.555 334.241 195.197 334.241 170.258V156.285C334.241 131.366 355.49 112.008 380.648 112.008C393.747 112.008 402.958 116.528 412.581 123.485C416.862 126.59 417.805 132.545 414.702 136.809C411.579 141.074 405.589 142.017 401.308 138.912C393.59 133.331 388.268 131.071 380.667 131.071L380.648 131.09ZM474.875 131.09C459.792 131.09 447.557 143.255 447.557 158.289V168.509C447.557 183.543 459.792 195.708 474.875 195.708C489.958 195.708 501.977 183.602 501.977 168.509V158.289C501.977 143.196 489.879 131.09 474.875 131.09ZM428.37 158.289C428.37 132.741 449.188 112.008 474.875 112.008C500.563 112.008 521.164 132.8 521.164 158.289V168.509C521.164 193.998 500.622 214.79 474.875 214.79C449.129 214.79 428.37 194.057 428.37 168.509V158.289ZM584.774 131.601C569.652 131.601 557.457 143.747 557.457 158.683C557.457 173.618 569.672 185.764 584.774 185.764C599.877 185.764 611.876 173.697 611.876 158.683C611.876 143.668 599.818 131.601 584.774 131.601ZM538.269 158.683C538.269 133.154 559.126 112.519 584.774 112.519C595.693 112.519 605.67 116.253 613.545 122.483L620.733 115.329C624.484 111.595 630.552 111.595 634.303 115.329C638.054 119.063 638.054 125.096 634.303 128.83L625.819 137.281C629.178 143.688 631.063 150.979 631.063 158.702C631.063 184.152 610.501 204.866 584.774 204.866C584.519 204.866 584.264 204.866 584.008 204.866H563.643C560.226 204.866 557.457 207.617 557.457 211.017C557.457 214.417 560.226 217.168 563.643 217.168H589.939H598.345C605.258 217.168 612.426 219.075 618.18 223.614C624.131 228.292 627.901 235.229 628.569 243.739C629.747 258.812 619.123 269.11 610.482 272.431L586.444 283.004C581.593 285.127 575.937 282.945 573.796 278.131C571.655 273.316 573.855 267.675 578.686 265.553L602.96 254.882C603.137 254.803 603.333 254.724 603.51 254.665C604.531 254.292 606.259 253.191 607.614 251.364C608.871 249.674 609.598 247.649 609.421 245.252C609.146 241.754 607.811 239.808 606.259 238.609C604.551 237.253 601.84 236.271 598.325 236.271H564.036C563.937 236.271 563.839 236.271 563.721 236.271H563.604C549.601 236.271 538.23 224.97 538.23 211.037C538.23 201.997 543.002 194.077 550.171 189.616C542.747 181.44 538.23 170.612 538.23 158.722L538.269 158.683ZM694.045 131.601C679.021 131.601 666.825 143.727 666.825 158.683V205.239C666.825 210.506 662.525 214.79 657.242 214.79C651.959 214.79 647.658 210.526 647.658 205.239V158.683C647.658 133.193 668.436 112.519 694.065 112.519C719.693 112.519 740.471 133.193 740.471 158.683V205.239C740.471 210.506 736.17 214.79 730.887 214.79C725.605 214.79 721.304 210.526 721.304 205.239V158.683C721.304 143.727 709.128 131.601 694.084 131.601H694.045ZM807.204 131.621C791.748 131.621 779.356 143.963 779.356 159.017V168.843C779.356 183.897 791.748 196.238 807.204 196.238C812.565 196.238 817.514 194.745 821.698 192.19C826.214 189.439 832.126 190.834 834.895 195.334C837.664 199.835 836.25 205.711 831.733 208.462C824.604 212.825 816.179 215.321 807.204 215.321C781.3 215.321 760.169 194.588 760.169 168.843V159.017C760.169 133.272 781.3 112.538 807.204 112.538C829.357 112.538 847.778 127.671 852.707 148.07L854.632 156.049L813.744 172.597C808.834 174.581 803.237 172.243 801.234 167.349C799.231 162.475 801.587 156.894 806.497 154.909L830.947 145.004C826.156 136.986 817.338 131.601 807.165 131.601L807.204 131.621ZM912.37 131.621C896.914 131.621 884.522 143.963 884.522 159.017V168.843C884.522 183.897 896.914 196.238 912.37 196.238C917.732 196.238 922.681 194.745 926.864 192.19C928.965 190.913 930.89 189.36 932.559 187.572C936.192 183.72 942.261 183.543 946.11 187.139C949.979 190.736 950.175 196.789 946.542 200.621C943.694 203.628 940.454 206.281 936.879 208.462C929.731 212.825 921.326 215.321 912.331 215.321C886.427 215.321 865.296 194.588 865.296 168.843V159.017C865.296 133.272 886.427 112.538 912.331 112.538C934.484 112.538 952.905 127.671 957.834 148.07L959.759 156.049L918.871 172.597C913.961 174.581 908.364 172.243 906.361 167.349C904.358 162.475 906.714 156.894 911.624 154.909L936.074 145.004C931.282 136.986 922.465 131.601 912.292 131.601L912.37 131.621Z" fill="#6510F4"/>
    #     </mask>
    #     <g mask="url(#mask0_103_2579)">
    #     <rect x="86" y="-119" width="1120" height="561" fill="#6510F4"/>
    #     </g>
    #     </svg>"""
    # Convert the SVG to a ReportLab Drawing

    logo_url = "data:image/png;base64,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"

    # Finally, add the PNG image to your Bokeh plot as an image_url
    p.image_url(
        url=[logo_url],
        x=-layout_scale * 0.5,
        y=layout_scale * 0.5,
        w=layout_scale,
        h=layout_scale,
        anchor=position,
        global_alpha=logo_alpha,
    )


def graph_to_tuple(graph):
    """
    Converts a networkx graph to a tuple of (nodes, edges).

    :param graph: A networkx graph.
    :return: A tuple (nodes, edges).
    """
    nodes = list(graph.nodes(data=True))  # Get nodes with attributes
    edges = list(graph.edges(data=True))  # Get edges with attributes
    return (nodes, edges)


def start_visualization_server(
    host="0.0.0.0", port=8001, handler_class=http.server.SimpleHTTPRequestHandler
):
    """
    Spin up a simple HTTP server in a background thread to serve files.
    This is especially handy for quick demos or visualization purposes.

    Returns a shutdown() function that can be called to stop the server.

    :param host: Host/IP to bind to. Defaults to '0.0.0.0'.
    :param port: Port to listen on. Defaults to 8001.
    :param handler_class: A handler class, defaults to SimpleHTTPRequestHandler.
    :return: A no-argument function `shutdown` which, when called, stops the server.
    """
    # Create the server
    server = socketserver.TCPServer((host, port), handler_class)

    def _serve_forever():
        print(f"Visualization server running at: http://{host}:{port}")
        server.serve_forever()

    # Start the server in a background thread
    thread = Thread(target=_serve_forever, daemon=True)
    thread.start()

    def shutdown():
        """
        Shuts down the server and blocks until the thread is joined.
        """
        server.shutdown()  # Signals the serve_forever() loop to stop
        server.server_close()  # Frees up the socket
        thread.join()
        print(f"Visualization server on port {port} has been shut down.")

    # Return only the shutdown function (the server runs in the background)
    return shutdown
