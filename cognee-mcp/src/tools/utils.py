"""
Utility functions for cognee tools.
"""

import os
import importlib.util


def node_to_string(node):
    """Convert a node dictionary to a string representation."""
    node_data = ", ".join(
        [f'{key}: "{value}"' for key, value in node.items() if key in ["id", "name"]]
    )
    return f"Node({node_data})"


def retrieved_edges_to_string(search_results):
    """Convert graph search results (triplets) to human-readable strings."""
    edge_strings = []
    for triplet in search_results:
        node1, edge, node2 = triplet
        relationship_type = edge["relationship_name"]
        edge_str = f"{node_to_string(node1)} {relationship_type} {node_to_string(node2)}"
        edge_strings.append(edge_str)
    return "\n".join(edge_strings)


def load_class(model_file, model_name):
    """Dynamically load a class from a file."""
    model_file = os.path.abspath(model_file)
    spec = importlib.util.spec_from_file_location("graph_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, model_name)
    return model_class
