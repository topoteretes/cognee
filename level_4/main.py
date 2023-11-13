# from marvin import ai_classifier
# marvin.settings.openai.api_key = os.environ.get("OPENAI_API_KEY")
from cognitive_architecture.database.graph_database.graph import Neo4jGraphDB
from cognitive_architecture.database.postgres.models.memory import MemoryModel

import os
from dotenv import load_dotenv

from level_4.cognitive_architecture.database.postgres.database_crud import session_scope
from cognitive_architecture.database.postgres.database import AsyncSessionLocal

import instructor
from openai import OpenAI

# Adds response_model to ChatCompletion
# Allows the return of Pydantic model rather than raw JSON
instructor.patch(OpenAI())
DEFAULT_PRESET = "promethai_chat"
preset_options = [DEFAULT_PRESET]
PROMETHAI_DIR = os.path.join(os.path.expanduser("~"), ".")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
from cognitive_architecture.config import Config

config = Config()
config.load()

print(config.model)
print(config.openai_key)


import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

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
    from cognitive_architecture.database.graph_database.graph import Neo4jGraphDB
    # Example initialization (replace with your actual connection details)
    neo4j_graph_db = Neo4jGraphDB(url='bolt://localhost:7687', username='neo4j', password='pleaseletmein')
    results = []
    for namespace in vectordb_namespace:
        update_result = neo4j_graph_db.update_document_node_with_namespace(user_id, namespace)
        results.append(update_result)
    return results




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
from cognitive_architecture.database.postgres.database_crud import fetch_job_id
import uuid
from cognitive_architecture.database.postgres.models.sessions import Session
from cognitive_architecture.database.postgres.models.operation import Operation
from cognitive_architecture.database.postgres.database_crud import session_scope, add_entity, update_entity, fetch_job_id
from cognitive_architecture.database.postgres.models.metadatas import MetaDatas
from cognitive_architecture.database.postgres.models.testset import TestSet
from cognitive_architecture.database.postgres.models.testoutput import TestOutput
from cognitive_architecture.database.postgres.models.docs import DocsModel
from cognitive_architecture.database.postgres.models.memory import MemoryModel
async def main():
    user_id = "user"

    async with session_scope(AsyncSessionLocal()) as session:
        # out = await get_vectordb_namespace(session, user_id)






        # print(out)

        # job_id = ""
        # job_id = await fetch_job_id(session, user_id=user_id, job_id=job_id)
        # if job_id is None:
        #     job_id = str(uuid.uuid4())
        #
        #     await add_entity(
        #         session,
        #         Operation(
        #             id=job_id,
        #             user_id=user_id,
        #             operation_params="",
        #             number_of_files=2,
        #             operation_status = "RUNNING",
        #             operation_type="",
        #             test_set_id="",
        #         ),
        #     )

        # await update_document_vectordb_namespace(session, user_id)
        # from cognitive_architecture.graph_database.graph import Neo4jGraphDB
        # # Example initialization (replace with your actual connection details)
        neo4j_graph_db = Neo4jGraphDB(url='bolt://localhost:7687', username='neo4j', password='pleaseletmein')
        # # Generate the Cypher query for a specific user
        # user_id = 'user123'  # Replace with the actual user ID
        cypher_query = await neo4j_graph_db.generate_cypher_query_for_user_prompt_decomposition(user_id,"I walked in the forest yesterday and added to my list I need to buy some milk in the store")
        # result = neo4j_graph_db.query(cypher_query)
        call_of_the_wild_summary = {
            "user_id": user_id,
            "document_category": "Classic Literature",
            "title": "The Call of the Wild",
            "summary": (
                "'The Call of the Wild' is a novel by Jack London set in the Yukon during the 1890s Klondike "
                "Gold Rush—a period when strong sled dogs were in high demand. The novel's central character "
                "is a dog named Buck, a domesticated dog living at a ranch in the Santa Clara Valley of California "
                "as the story opens. Stolen from his home and sold into the brutal existence of an Alaskan sled dog, "
                "he reverts to atavistic traits. Buck is forced to adjust to, and survive, cruel treatments and fight "
                "to dominate other dogs in a harsh climate. Eventually, he sheds the veneer of civilization, relying "
                "on primordial instincts and lessons he learns, to emerge as a leader in the wild. London drew on his "
                "own experiences in the Klondike, and the book provides a snapshot of the epical gold rush and the "
                "harsh realities of life in the wilderness. The novel explores themes of morality versus instinct, "
                "the struggle for survival in the natural world, and the intrusion of civilization on the wilderness. "
                "As Buck's wild nature is awakened, he rises to become a respected and feared leader in the wild, "
                "answering the primal call of nature."
            )
        }
        rs = neo4j_graph_db.create_document_node_cypher(call_of_the_wild_summary, user_id)

        neo4j_graph_db.query(rs, call_of_the_wild_summary)
        print(cypher_query)

        neo4j_graph_db.update_document_node_with_namespace(user_id, document_title="The Call of the Wild")



        # await update_document_vectordb_namespace(session, user_id)
        # # Execute the generated Cypher query
        # result = neo4j_graph_db.query(cypher_query)



        params = {
            "version": "1.0",
            "agreement_id": "AG123456",
            "privacy_policy": "https://example.com/privacy",
            "terms_of_service": "https://example.com/terms",
            "format": "json",
            "schema_version": "1.1",
            "checksum": "a1b2c3d4e5f6",
            "owner": "John Doe",
            "license": "MIT",
            "validity_start": "2023-08-01",
            "validity_end": "2024-07-31",
        }
        loader_settings = {
            "format": "PDF",
            "source": "URL",
            "path": "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf",
        }
        # memory_instance = Memory(namespace='SEMANTICMEMORY')
        # sss = await memory_instance.dynamic_method_call(memory_instance.semantic_memory_class, 'fetch_memories', observation='some_observation')
        # from cognitive_architecture.vectorstore_manager import Memory
        #
        #
        # memory = await Memory.create_memory("676", session, namespace="SEMANTICMEMORY")
        #
        # # Adding a memory instance
        # await memory.add_memory_instance("ExampleMemory")
        #
        # # Managing memory attributes
        # existing_user = await Memory.check_existing_user("676", session)
        # print("here is the existing user", existing_user)
        # await memory.manage_memory_attributes(existing_user)
        # # aeehuvyq_semanticememory_class
        #
        # await memory.add_dynamic_memory_class("semanticmemory", "SEMANTICMEMORY")
        # await memory.add_method_to_class(memory.semanticmemory_class, "add_memories")
        # # await memory.add_method_to_class(memory.semanticmemory_class, "fetch_memories")
        # sss = await memory.dynamic_method_call(memory.semanticmemory_class, 'add_memories',
        #                                                 observation='some_observation', params=params, loader_settings=loader_settings)

    # print(rs)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


