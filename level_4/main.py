from enum import Enum

import typer
import os
import uuid
# import marvin
# from pydantic_settings import BaseSettings
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
# from marvin import ai_classifier
# marvin.settings.openai.api_key = os.environ.get("OPENAI_API_KEY")

from cognitive_architecture.models.sessions import Session
from cognitive_architecture.models.testset import TestSet
from cognitive_architecture.models.testoutput import TestOutput
from cognitive_architecture.models.metadatas import MetaDatas
from cognitive_architecture.models.operation import Operation
from cognitive_architecture.models.docs import DocsModel
from cognitive_architecture.models.memory import MemoryModel

from pathlib import Path

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Neo4jVector
import os
from dotenv import load_dotenv
import uuid

from graphviz import Digraph

from cognitive_architecture.database.database_crud import session_scope
from cognitive_architecture.database.database import AsyncSessionLocal

import openai
import instructor

# Adds response_model to ChatCompletion
# Allows the return of Pydantic model rather than raw JSON
instructor.patch()
from pydantic import BaseModel, Field
from typing import List
DEFAULT_PRESET = "promethai_chat"
preset_options = [DEFAULT_PRESET]
import questionary
PROMETHAI_DIR = os.path.join(os.path.expanduser("~"), ".")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
from cognitive_architecture.config import Config

config = Config()
config.load()

print(config.model)
print(config.openai_key)


import logging









import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

async def get_vectordb_namespace(session: AsyncSession, user_id: str):
    try:
        result = await session.execute(
            select(MemoryModel.id).where(MemoryModel.user_id == user_id).order_by(MemoryModel.created_at.desc()).limit(1)
        )
        namespace = result.scalar_one_or_none()
        return namespace
    except Exception as e:
        logging.error(f"An error occurred while retrieving the Vectordb_namespace: {str(e)}")
        return None

# async def retrieve_job_by_id(session, user_id, job_id):
#     try:
#         result = await session.execute(
#             session.query(Session.id)
#             .filter_by(user_id=user_id, id=job_id)
#             .order_by(Session.created_at)
#         )
#         return result.scalar_one_or_none()
#     except Exception as e:
#         logging.error(f"An error occurred while retrieving the job: {str(e)}")
#         return None




async def update_document_vectordb_namespace(postgres_session: AsyncSession, user_id: str, namespace: str = None):
    """
    Update the Document node with the Vectordb_namespace for the given user. If the namespace is not provided,
    it will be retrieved from the PostgreSQL database.

    Args:
    postgres_session (AsyncSession): The async session for connecting to the PostgreSQL database.
    user_id (str): The user's unique identifier.
    namespace (str, optional): The Vectordb_namespace. If None, it will be retrieved from the database.

    Returns:
    The result of the update operation or None if an error occurred.
    """
    vectordb_namespace = namespace

    # Retrieve namespace from the database if not provided
    if vectordb_namespace is None:
        vectordb_namespace = await get_vectordb_namespace(postgres_session, user_id)
        if not vectordb_namespace:
            logging.error("Vectordb_namespace could not be retrieved.")
            return None

    # Update the Document node in Neo4j with the namespace
    update_result = update_document_node_with_namespace(user_id, vectordb_namespace)
    return update_result




    # query_input = "I walked in the forest yesterday and added to my list I need to buy some milk in the store"
    #
    # # Generate the knowledge graph from the user input
    # knowledge_graph = generate_graph(query_input)
    # visualize_knowledge_graph(knowledge_graph)
    # # out = knowledge_graph.dict()
    # # print(out)
    # #
    # graph: KnowledgeGraph = generate_graph("I walked in the forest yesterday and added to my list I need to buy some milk in the store")
    # graph_dic = graph.dict()
    #
    # node_variable_mapping = create_node_variable_mapping(graph_dic['nodes'])
    # edge_variable_mapping = create_edge_variable_mapping(graph_dic['edges'])
    # # Create unique variable names for each node
    # unique_node_variable_mapping = append_uuid_to_variable_names(node_variable_mapping)
    # unique_edge_variable_mapping = append_uuid_to_variable_names(edge_variable_mapping)
    # create_nodes_statements = generate_create_statements_for_nodes_with_uuid(graph_dic['nodes'], unique_node_variable_mapping)
    # create_edges_statements = generate_create_statements_for_edges_with_uuid(graph_dic['edges'], unique_node_variable_mapping)
    #
    # memory_type_statements_with_uuid_and_time_context = generate_memory_type_relationships_with_uuid_and_time_context(
    #     graph_dic['nodes'], unique_node_variable_mapping)
    #
    # # # Combine all statements
    # cypher_statements = [create_base_queries_from_user(user_id)] + create_nodes_statements + create_edges_statements + memory_type_statements_with_uuid_and_time_context
    # cypher_statements_joined = "\n".join(cypher_statements)
    #
    #
    #
    # execute_cypher_query(cypher_statements_joined)

    # bartleby_summary = {
    #     "document_category": "Classic Literature",
    #     "title": "Bartleby, the Scrivener",
    #     "summary": (
    #         "Bartleby, the Scrivener: A Story of Wall Street' is a short story by Herman Melville "
    #         "that tells the tale of Bartleby, a scrivener, or copyist, who works for a Manhattan "
    #         "lawyer. Initially, Bartleby is a competent and industrious worker. However, one day, "
    #         "when asked to proofread a document, he responds with what becomes his constant refrain "
    #         "to any request: 'I would prefer not to.' As the story progresses, Bartleby becomes "
    #         "increasingly passive, refusing not just work but also food and eventually life itself, "
    #         "as he spirals into a state of passive resistance. The lawyer, the narrator of the story, "
    #         "is both fascinated and frustrated by Bartleby's behavior. Despite attempts to understand "
    #         "and help him, Bartleby remains an enigmatic figure, his motives and thoughts unexplained. "
    #         "He is eventually evicted from the office and later found dead in a prison yard, having "
    #         "preferred not to live. The story is a meditation on the themes of isolation, societal "
    #         "obligation, and the inexplicable nature of human behavior."
    #     )
    # }
    # rs = create_document_node_cypher(bartleby_summary, user_id)
    #
    # parameters = {
    #     'user_id': user_id,
    #     'title': bartleby_summary['title'],
    #     'summary': bartleby_summary['summary'],
    #     'document_category': bartleby_summary['document_category']
    # }
    #
    # execute_cypher_query(rs, parameters)
#
# async def main():
#     user_id = "User1"
#
#     async with session_scope(AsyncSessionLocal()) as session:
#         await update_document_vectordb_namespace(session, user_id)
#
#     # print(rs)
#
# if __name__ == "__main__":
#     import asyncio
#
#     asyncio.run(main())
#
#     # config = Config()
#     # config.load()
#     #
#     # print(config.model)
#     # print(config.openai_key)



async def main():
    user_id = "User1"
    from cognitive_architecture.graph_database.graph import Neo4jGraphDB
    # Example initialization (replace with your actual connection details)
    neo4j_graph_db = Neo4jGraphDB(url='bolt://localhost:7687', username='neo4j', password='pleaseletmein')
    # Generate the Cypher query for a specific user
    user_id = 'user123'  # Replace with the actual user ID
    cypher_query = neo4j_graph_db.generate_cypher_query_for_user_prompt_decomposition(user_id)
    # Execute the generated Cypher query
    result = neo4j_graph_db.query(cypher_query)

    # async with session_scope(AsyncSessionLocal()) as session:
    #     await update_document_vectordb_namespace(session, user_id)

    # print(rs)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())