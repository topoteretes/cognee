import os
import random
import string
import uuid

from graphviz import Digraph

# from graph_database.graph import KnowledgeGraph


class Node:
    def __init__(self, id, description, color):
        self.id = id
        self.description = description
        self.color = color

class Edge:
    def __init__(self, source, target, label, color):
        self.source = source
        self.target = target
        self.label = label
        self.color = color
# def visualize_knowledge_graph(kg: KnowledgeGraph):
#     dot = Digraph(comment="Knowledge Graph")
#
#     # Add nodes
#     for node in kg.nodes:
#         dot.node(str(node.id), node.description, color=node.color)
#
#     # Add edges
#     for edge in kg.edges:
#         dot.edge(str(edge.source), str(edge.target), label=edge.description, color=edge.color)
#
#     # Render the graph
#     dot.render("knowledge_graph.gv", view=True)
#
#
def get_document_names(doc_input):
    """
    Get a list of document names.

    This function takes doc_input, which can be a folder path, a single document file path, or a document name as a string.
    It returns a list of document names based on the doc_input.

    Args:
        doc_input (str): The doc_input can be a folder path, a single document file path, or a document name as a string.

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
    # Initialize an empty list to store formatted items
    formatted_items = []

    # Iterate through all key-value pairs
    for key, value in d.items():
        # Format key-value pairs with a colon and space, and adding quotes for string values
        formatted_item = f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}"
        formatted_items.append(formatted_item)

    # Join all formatted items with a comma and a space
    formatted_string = ", ".join(formatted_items)

    # Add curly braces to mimic a dictionary
    formatted_string = f"{{{formatted_string}}}"

    return formatted_string


def append_uuid_to_variable_names(variable_mapping):
    unique_variable_mapping = {}
    for original_name in variable_mapping.values():
        unique_name = f"{original_name}_{uuid.uuid4().hex}"
        unique_variable_mapping[original_name] = unique_name
    return unique_variable_mapping


# Update the functions to use the unique variable names
def create_node_variable_mapping(nodes):
    mapping = {}
    for node in nodes:
        variable_name = f"{node['category']}{node['id']}".lower()
        mapping[node['id']] = variable_name
    return mapping


def create_edge_variable_mapping(edges):
    mapping = {}
    for edge in edges:
        # Construct a unique identifier for the edge
        variable_name = f"edge{edge['source']}to{edge['target']}".lower()
        mapping[(edge['source'], edge['target'])] = variable_name
    return mapping



def generate_letter_uuid(length=8):
    """Generate a random string of uppercase letters with the specified length."""
    letters = string.ascii_uppercase  # A-Z
    return "".join(random.choice(letters) for _ in range(length))


