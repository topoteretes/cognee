import os
import random
import string
import uuid

from graphviz import Digraph
from sqlalchemy import or_
from sqlalchemy.orm import contains_eager

from cognitive_architecture.database.postgres.database import AsyncSessionLocal
from dotenv import load_dotenv

load_dotenv()


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

from cognitive_architecture.database.postgres.models.operation import Operation
from cognitive_architecture.database.postgres.database_crud import session_scope, add_entity, update_entity, fetch_job_id
from cognitive_architecture.database.postgres.models.metadatas import MetaDatas
from cognitive_architecture.database.postgres.models.docs import DocsModel
from cognitive_architecture.database.postgres.models.memory import MemoryModel
from cognitive_architecture.database.postgres.models.user import User
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import logging
async def get_vectordb_namespace(session: AsyncSession, user_id: str):
    try:
        result = await session.execute(
            select(MemoryModel.memory_name).where(MemoryModel.user_id == user_id).order_by(MemoryModel.created_at.desc())
        )
        namespace = [row[0] for row in result.fetchall()]
        return namespace
    except Exception as e:
        logging.error(f"An error occurred while retrieving the Vectordb_namespace: {str(e)}")
        return None

async def get_vectordb_document_name(session: AsyncSession, user_id: str):
    try:
        result = await session.execute(
            select(DocsModel.doc_name).where(DocsModel.user_id == user_id).order_by(DocsModel.created_at.desc())
        )
        doc_names = [row[0] for row in result.fetchall()]
        return doc_names
    except Exception as e:
        logging.error(f"An error occurred while retrieving the Vectordb_namespace: {str(e)}")
        return None


async def get_model_id_name(session: AsyncSession, id: str):
    try:
        result = await session.execute(
            select(MemoryModel.memory_name).where(MemoryModel.id == id).order_by(MemoryModel.created_at.desc())
        )
        doc_names = [row[0] for row in result.fetchall()]
        return doc_names
    except Exception as e:
        logging.error(f"An error occurred while retrieving the Vectordb_namespace: {str(e)}")
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
            contains_eager(Operation.memories)  # Informs ORM of the join for memories
        )
        .where(
            (Operation.user_id == user_id) &  # Filter by user_id
            or_(
                DocsModel.graph_summary == False,  # Condition 1: graph_summary is False
                DocsModel.graph_summary == None  # Condition 3: graph_summary is None
            )  # Filter by user_id
        )
        .order_by(Operation.created_at.desc())  # Order by creation date
    )

    operations = result.unique().scalars().all()

    # Extract memory names and document names and IDs
    # memory_names = [memory.memory_name for op in operations for memory in op.memories]
    memory_details = [(memory.memory_name, memory.memory_category) for op in operations for memory in op.memories]
    docs = [(doc.doc_name, doc.id) for op in operations for doc in op.docs]

    return memory_details, docs

    # except Exception as e:
    #     # Handle the exception as needed
    #     print(f"An error occurred: {e}")
    #     return None
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
            .join(Operation, Operation.id == MemoryModel.operation_id)  # Join with Operation
            .join(DocsModel, DocsModel.operation_id == Operation.id)  # Join with DocsModel
            .where(DocsModel.id == docs_id)  # Filtering based on the passed document ID
            .distinct()  # To avoid duplicate memory names
        )

        memory_names = [row[0] for row in result.fetchall()]
        return memory_names

    except Exception as e:
        # Handle the exception as needed
        print(f"An error occurred: {e}")
        return None



#
# async def main():
#     user_id = "user"
#
#     async with session_scope(AsyncSessionLocal()) as session:
#         output = await get_unsumarized_vector_db_namespace(session, user_id)
#
#         print(output)
#         # await update_entity(session, DocsModel, "8cd9a022-5a7a-4af5-815a-f988415536ae", True)
#         # out = await get_vectordb_namespace(session, user_id)
#         # params = {
#         #     "version": "1.0",
#         #     "agreement_id": "AG123456",
#         #     "privacy_policy": "https://example.com/privacy",
#         #     "terms_of_service": "https://example.com/terms",
#         #     "format": "json",
#         #     "schema_version": "1.1",
#         #     "checksum": "a1b2c3d4e5f6",
#         #     "owner": "John Doe",
#         #     "license": "MIT",
#         #     "validity_start": "2023-08-01",
#         #     "validity_end": "2024-07-31",
#         # }
#         # loader_settings = {
#         #     "format": "PDF",
#         #     "source": "DEVICE",
#         #     "path": [".data"],
#         #     "strategy": "SUMMARY",
#         # }
#         # await load_documents_to_vectorstore(session, user_id, loader_settings=loader_settings)
#         # await user_query_to_graph_db(session, user_id, "I walked in the forest yesterday and added to my list I need to buy some milk in the store and get a summary from a classical book i read yesterday")
#         # await add_documents_to_graph_db(session, user_id, loader_settings=loader_settings)
#         # await user_context_enrichment(session, user_id, query="Tell me about the book I read yesterday")
#
#
# if __name__ == "__main__":
#     import asyncio
#
#     asyncio.run(main())
