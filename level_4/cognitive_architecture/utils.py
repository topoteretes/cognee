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