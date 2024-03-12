""" This module contains utility functions for the cognitive architecture. """

import os
import uuid
import random
import string
import logging
import graphistry
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import or_
from sqlalchemy.future import select
from sqlalchemy.orm import contains_eager
from sqlalchemy.ext.asyncio import AsyncSession
from cognitive_architecture.database.relationaldb.models.docs import DocsModel
from cognitive_architecture.database.relationaldb.models.memory import MemoryModel
from cognitive_architecture.database.relationaldb.models.operation import Operation

from cognitive_architecture.config import Config

config = Config()
config.load()

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
    """ Format a dictionary as a string."""
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


def append_uuid_to_variable_names(variable_mapping):
    """ Append a UUID to the variable names to make them unique."""
    unique_variable_mapping = {}
    for original_name in variable_mapping.values():
        unique_name = f"{original_name}_{uuid.uuid4().hex}"
        unique_variable_mapping[original_name] = unique_name
    return unique_variable_mapping


# Update the functions to use the unique variable names
def create_node_variable_mapping(nodes):
    """ Create a mapping of node identifiers to unique variable names."""
    mapping = {}
    for node in nodes:
        variable_name = f"{node['category']}{node['id']}".lower()
        mapping[node["id"]] = variable_name
    return mapping


def create_edge_variable_mapping(edges):
    """ Create a mapping of edge identifiers to unique variable names."""
    mapping = {}
    for edge in edges:
        # Construct a unique identifier for the edge
        variable_name = f"edge{edge['source']}to{edge['target']}".lower()
        mapping[(edge["source"], edge["target"])] = variable_name
    return mapping


def generate_letter_uuid(length=8):
    """Generate a random string of uppercase letters with the specified length."""
    letters = string.ascii_uppercase  # A-Z
    return "".join(random.choice(letters) for _ in range(length))




async def get_vectordb_namespace(session: AsyncSession, user_id: str):
    """ Asynchronously retrieves the latest memory names for a given user."""
    try:
        result = await session.execute(
            select(MemoryModel.memory_name)
            .where(MemoryModel.user_id == user_id)
            .order_by(MemoryModel.created_at.desc())
        )
        namespace = [row[0] for row in result.fetchall()]
        return namespace
    except Exception as e:
        logging.error(
            f"An error occurred while retrieving the Vectordb_namespace: {str(e)}"
        )
        return None


async def get_vectordb_document_name(session: AsyncSession, user_id: str):
    """ Asynchronously retrieves the latest memory names for a given user."""
    try:
        result = await session.execute(
            select(DocsModel.doc_name)
            .where(DocsModel.user_id == user_id)
            .order_by(DocsModel.created_at.desc())
        )
        doc_names = [row[0] for row in result.fetchall()]
        return doc_names
    except Exception as e:
        logging.error(
            f"An error occurred while retrieving the Vectordb_namespace: {str(e)}"
        )
        return None


async def get_model_id_name(session: AsyncSession, id: str):
    """ Asynchronously retrieves the latest memory names for a given user."""
    try:
        result = await session.execute(
            select(MemoryModel.memory_name)
            .where(MemoryModel.id == id)
            .order_by(MemoryModel.created_at.desc())
        )
        doc_names = [row[0] for row in result.fetchall()]
        return doc_names
    except Exception as e:
        logging.error(
            f"An error occurred while retrieving the Vectordb_namespace: {str(e)}"
        )
        return None


async def get_unsumarized_vector_db_namespace(session: AsyncSession, user_id: str):
    """
    Asynchronously retrieves the latest memory names and document details for a given user.

    This function executes a database query to fetch memory names and document details
    associated with operations performed by a specific user. It leverages explicit joins
    with the 'docs' and 'memories' tables and applies eager loading to optimize performance.

    Parameters:
    - session (AsyncSession): The database session for executing the query.
    - user_id (str): The unique identifier of the user.

    Returns:
    - Tuple[List[str], List[Tuple[str, str]]]: A tuple containing a list of memory names and
      a list of tuples with document names and their corresponding IDs.
      Returns None if an exception occurs.

    Raises:
    - Exception: Propagates any exceptions that occur during query execution.

    Example Usage:
    """
    # try:
    result = await session.execute(
        select(Operation)
        .join(Operation.docs)  # Explicit join with docs table
        .join(Operation.memories)  # Explicit join with memories table
        .options(
            contains_eager(Operation.docs),  # Informs ORM of the join for docs
            contains_eager(Operation.memories),  # Informs ORM of the join for memories
        )
        .where(
            (Operation.user_id == user_id)
            & or_(  # Filter by user_id
                DocsModel.graph_summary == False,  # Condition 1: graph_summary is False
                DocsModel.graph_summary == None,  # Condition 3: graph_summary is None
            )  # Filter by user_id
        )
        .order_by(Operation.created_at.desc())  # Order by creation date
    )

    operations = result.unique().scalars().all()

    # Extract memory names and document names and IDs
    # memory_names = [memory.memory_name for op in operations for memory in op.memories]
    memory_details = [
        (memory.memory_name, memory.memory_category)
        for op in operations
        for memory in op.memories
    ]
    docs = [(doc.doc_name, doc.id) for op in operations for doc in op.docs]

    return memory_details, docs

async def get_memory_name_by_doc_id(session: AsyncSession, docs_id: str):
    """
    Asynchronously retrieves memory names associated with a specific document ID.

    This function executes a database query to fetch memory names linked to a document
    through operations. The query is filtered based on a given document ID and retrieves
    only the memory names without loading the entire Operation entity.

    Parameters:
    - session (AsyncSession): The database session for executing the query.
    - docs_id (str): The unique identifier of the document.

    Returns:
    - List[str]: A list of memory names associated with the given document ID.
      Returns None if an exception occurs.

    Raises:
    - Exception: Propagates any exceptions that occur during query execution.
    """
    try:
        result = await session.execute(
            select(MemoryModel.memory_name)
            .join(
                Operation, Operation.id == MemoryModel.operation_id
            )  # Join with Operation
            .join(
                DocsModel, DocsModel.operation_id == Operation.id
            )  # Join with DocsModel
            .where(DocsModel.id == docs_id)  # Filtering based on the passed document ID
            .distinct()  # To avoid duplicate memory names
        )

        memory_names = [row[0] for row in result.fetchall()]
        return memory_names

    except Exception as e:
        # Handle the exception as needed
        print(f"An error occurred: {e}")
        return None


async def read_query_prompt(filename: str) -> str:
    """Read a query prompt from a file.
    :param filename: The name of the file to read.
    :return: The content of the file as a string.
    """
    script_directory = Path(__file__).parent

    # Set the base directory relative to the script's directory
    base_directory = script_directory.parent / "cognitive_architecture/infrastructure/llm/prompts"

    # Construct the full file path
    file_path = base_directory / filename
    try:
        return file_path.read_text()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path.absolute()}")
    except Exception as e:
        logging.error(f"An error of type {type(e).__name__} occurred while reading file: {file_path.absolute()}. Error message: {e}")
    return None



async def print_file_content(file_path):
    # Create a Path object for the file path
    path = Path(file_path)

    # Check if the file exists
    if path.is_file():
        # Open and read the file, then print its content
        with path.open('r') as file:
            print(file.read())
    else:
        # Print an error message if the file does not exist
        print(f"The file '{file_path}' does not exist.")

async def async_render_template(filename: str,  context: dict) -> str:
    """Render a Jinja2 template asynchronously.
    :param filename: The name of the template file to render.
    :param context: The context to render the template with.
    :return: The rendered template as a string."""
    # Initialize the Jinja2 environment to load templates from the filesystem
    script_directory = Path(__file__).parent

    # Set the base directory relative to the script's directory
    base_directory = script_directory.parent / "cognitive_architecture/infrastructure/llm/prompts"


    # Construct the full file path
    file_path = base_directory / filename

    env = Environment(
        loader=FileSystemLoader(base_directory),
        autoescape=select_autoescape(['html', 'xml', 'txt'])
    )

    # Load the template by name
    template = env.get_template(filename)

    # Render the template with the provided context
    rendered_template = template.render(context)

    return rendered_template


async def render_graph(graph, graph_type):

    # Authenticate with your Graphistry API key

    import networkx as nx
    import pandas as pd

    graphistry.register(api=3, username=config.graphistry_username, password=config.graphistry_password)
    # Convert the NetworkX graph to a Pandas DataFrame representing the edge list
    edges = nx.to_pandas_edgelist(graph)
    # Visualize the graph using Graphistry
    plotter = graphistry.edges(edges, 'source', 'target')
    # Visualize the graph (this will open a URL in your default web browser)
    url = plotter.plot(render=False, as_files=True)
    print(f"Graph is visualized at: {url}")


# import networkx as nx
# # Create a simple NetworkX graph
# G = nx.Graph()
#
# # Add nodes
# G.add_node(1)
# G.add_node(2)
#
# # Add an edge between nodes
# G.add_edge(1, 2)
#
# import asyncio
#
# # Define the graph type (for this example, it's just a placeholder as the function doesn't use it yet)
# graph_type = "simple"
#
# # Call the render_graph function
# asyncio.run(render_graph(G, graph_type))

