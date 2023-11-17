# from marvin import ai_classifier
# marvin.settings.openai.api_key = os.environ.get("OPENAI_API_KEY")
from pydantic import BaseModel

from cognitive_architecture.database.graph_database.graph import Neo4jGraphDB
from cognitive_architecture.database.postgres.models.memory import MemoryModel
from cognitive_architecture.classifiers.classifier import classify_documents
import os
from dotenv import load_dotenv

from cognitive_architecture.database.postgres.database_crud import session_scope
from cognitive_architecture.database.postgres.database import AsyncSessionLocal
from cognitive_architecture.utils import generate_letter_uuid
import instructor
from openai import OpenAI

from cognitive_architecture.vectorstore_manager import Memory
from cognitive_architecture.database.postgres.database_crud import fetch_job_id
import uuid
from cognitive_architecture.database.postgres.models.sessions import Session
from cognitive_architecture.database.postgres.models.operation import Operation
from cognitive_architecture.database.postgres.database_crud import session_scope, add_entity, update_entity, fetch_job_id
from cognitive_architecture.database.postgres.models.metadatas import MetaDatas

from cognitive_architecture.database.postgres.models.docs import DocsModel
from cognitive_architecture.database.postgres.models.memory import MemoryModel

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
from cognitive_architecture.utils import get_document_names

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

from sqlalchemy.orm import selectinload, joinedload


async def get_vectordb_data(session: AsyncSession, user_id: str):
    try:
        result = await session.execute(
            select(Operation, DocsModel.doc_name, DocsModel.id, MemoryModel.memory_name)
            .join(DocsModel, Operation.docs)  # Correct join with docs table
            .join(MemoryModel, Operation.memories)  # Correct join with memories table
            .where(
                (Operation.user_id == user_id)  # Filter by user_id
                # (Operation.operation_status == 'SUCCESS')  # Optional additional filter
            )
            .order_by(Operation.created_at.desc())  # Order by creation date
        )

        operations = result.scalars().all()

        # Extract memory names and document names and IDs
        memory_names = [memory.memory_name for op in operations for memory in op.memories]
        docs = [(doc.doc_name, doc.id) for op in operations for doc in op.docs]

        return memory_names, docs

    except Exception as e:
        # Handle the exception as needed
        print(f"An error occurred: {e}")
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




# async def update_document_vectordb_namespace(postgres_session: AsyncSession, user_id: str, namespace: str = None, job_id:str=None):
#     """
#     Update the Document node with the Vectordb_namespace for the given user. If the namespace is not provided,
#     it will be retrieved from the PostgreSQL database.
#
#     Args:
#     postgres_session (AsyncSession): The async session for connecting to the PostgreSQL database.
#     user_id (str): The user's unique identifier.
#     namespace (str, optional): The Vectordb_namespace. If None, it will be retrieved from the database.
#
#     Returns:
#     The result of the update operation or None if an error occurred.
#     """
#     vectordb_namespace = namespace
#
#     # Retrieve namespace from the database if not provided
#     if vectordb_namespace is None:
#         vectordb_namespace = await get_vectordb_namespace(postgres_session, user_id)
#         if not vectordb_namespace:
#             logging.error("Vectordb_namespace could not be retrieved.")
#             return None
#     from cognitive_architecture.database.graph_database.graph import Neo4jGraphDB
#     # Example initialization (replace with your actual connection details)
#     neo4j_graph_db = Neo4jGraphDB(url='bolt://localhost:7687', username='neo4j', password='pleaseletmein')
#     results = []
#     for namespace in vectordb_namespace:
#         update_result = neo4j_graph_db.update_document_node_with_namespace(user_id, namespace)
#         results.append(update_result)
#     return results



async def load_documents_to_vectorstore(session: AsyncSession, user_id: str, job_id:str=None, loader_settings:dict=None):
    namespace_id = str(generate_letter_uuid()) + "_" + "SEMANTICMEMORY"
    namespace_class = namespace_id + "_class"
    if job_id is None:
        job_id = str(uuid.uuid4())

    memory = await Memory.create_memory(user_id, session, namespace=namespace_id, job_id=job_id, memory_label=namespace_id)

    await add_entity(
        session,
        Operation(
            id=job_id,
            user_id=user_id,
            operation_status="RUNNING",
            operation_type="DATA_LOAD",
        ),
    )


    document_names = get_document_names(loader_settings.get("path", "None"))
    for doc in document_names:
        await add_entity(
            session,
            DocsModel(
                id=str(uuid.uuid4()),
                operation_id=job_id,
                doc_name=doc
            )
        )
        # Managing memory attributes
        existing_user = await Memory.check_existing_user(user_id, session)
        print("here is the existing user", existing_user)
        await memory.manage_memory_attributes(existing_user)
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
        print("Namespace id is %s", namespace_id)
        await memory.add_dynamic_memory_class(namespace_id.lower(), namespace_id)

        dynamic_memory_class = getattr(memory, namespace_class.lower(), None)

        methods_to_add = ["add_memories", "fetch_memories", "delete_memories"]

        if dynamic_memory_class is not None:
            for method_name in methods_to_add:
                await memory.add_method_to_class(dynamic_memory_class, method_name)
                print(f"Memory method {method_name} has been added")
        else:
            print(f"No attribute named  in memory.")

        print("Available memory classes:", await memory.list_memory_classes())
        result = await memory.dynamic_method_call(dynamic_memory_class, 'add_memories',
                                                        observation='some_observation', params=params, loader_settings=loader_settings)

        await update_entity(session, Operation, job_id, "SUCCESS")
        return result


async def user_query_to_graph_db(session: AsyncSession, user_id: str, query_input: str):

    job_id = str(uuid.uuid4())

    await add_entity(
        session,
        Operation(
            id=job_id,
            user_id=user_id,
            operation_status="RUNNING",
            operation_type="USER_QUERY_TO_GRAPH_DB",
        ),
    )

    neo4j_graph_db = Neo4jGraphDB(url=config.graph_database_url, username=config.graph_database_username, password=config.graph_database_password)
    # # Generate the Cypher query for a specific user
    # user_id = 'user123'  # Replace with the actual user ID
    cypher_query = await neo4j_graph_db.generate_cypher_query_for_user_prompt_decomposition(user_id,query_input)
    result = neo4j_graph_db.query(cypher_query)

    await update_entity(session, Operation, job_id, "SUCCESS")

    return result




async def add_documents_to_graph_db(postgres_session: AsyncSession, user_id: str):
    """"""
    try:
        # await update_document_vectordb_namespace(postgres_session, user_id)
        memory_names, docs = await get_vectordb_data(postgres_session, user_id)
        for doc, memory_name in zip(docs, memory_names):
            doc_name, doc_id = doc
            classification = await classify_documents(doc_name, document_id=doc_id)
            neo4j_graph_db = Neo4jGraphDB(url=config.graph_database_url, username=config.graph_database_username,
                                          password=config.graph_database_password)
            rs = neo4j_graph_db.create_document_node_cypher(classification, user_id)
            neo4j_graph_db.query(rs, classification)

            # select doc from the store
            neo4j_graph_db.update_document_node_with_namespace(user_id, vectordb_namespace=memory_name, document_id=doc_id)

    except:
        pass


async def user_context_enrichment(session, user_id, query):
    """"""
    neo4j_graph_db = Neo4jGraphDB(url=config.graph_database_url, username=config.graph_database_username,
                                  password=config.graph_database_password)

    semantic_mem = neo4j_graph_db.retrieve_semantic_memory(user_id=user_id)
    episodic_mem = neo4j_graph_db.retrieve_episodic_memory(user_id=user_id)
    context = f""" You are a memory system that uses cognitive architecture to enrich the user context.
    You have access to the following information:
    EPISODIC MEMORY: {episodic_mem}
    SEMANTIC MEMORY: {semantic_mem}
    PROCEDURAL MEMORY: NULL
    The original user query: {query}
    """

    classifier_prompt = """ Your task is to determine if any of the following document categories are relevant to the user query and context: {query} Context is: {context}
    Document categories: {document_categories}"""
    from cognitive_architecture.classifiers.classifier import classify_call

    document_categories = neo4j_graph_db.get_document_categories(user_id=user_id)
    relevant_categories = await classify_call( query, context, document_types=document_categories)
    fetch_namespace_from_graph = neo4j_graph_db.get_namespaces_by_document_category(user_id=user_id, category=relevant_categories)

    results = []

    for namespace in fetch_namespace_from_graph:
        memory = await Memory.create_memory(user_id, session, namespace=namespace)

        # Managing memory attributes
        existing_user = await Memory.check_existing_user(user_id, session)
        print("here is the existing user", existing_user)
        await memory.manage_memory_attributes(existing_user)
        namespace_class = namespace
        memory = await Memory.create_memory(user_id, session, namespace=namespace_class)

        # Managing memory attributes
        existing_user = await Memory.check_existing_user(user_id, session)
        print("here is the existing user", existing_user)
        await memory.manage_memory_attributes(existing_user)


        dynamic_memory_class = getattr(memory, namespace_class.lower(), None)

        await memory.add_dynamic_memory_class(dynamic_memory_class, namespace_class)
        await memory.add_method_to_class(dynamic_memory_class, "fetch_memories")
        raw_document_output = await memory.dynamic_method_call(
            memory.semanticmemory_class,
            "fetch_memories",
            observation=context,
        )
        from openai import OpenAI
        import instructor

        # Enables `response_model`
        client = instructor.patch(OpenAI())

        format_query_via_gpt = f""" Provide an answer to the user query: {query} Context is: {context}, Document store information is {raw_document_output} """

        class UserResponse(BaseModel):
            response: str


        user = client.chat.completions.create(
            model=config.model,
            response_model=UserResponse,
            messages=[
                {"role": "user", "content": format_query_via_gpt},
            ]
        )

        results.append(user.response)

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
        # neo4j_graph_db = Neo4jGraphDB(url='bolt://localhost:7687', username='neo4j', password='pleaseletmein')
        # # Generate the Cypher query for a specific user
        # user_id = 'user123'  # Replace with the actual user ID
        #cypher_query = await neo4j_graph_db.generate_cypher_query_for_user_prompt_decomposition(user_id,"I walked in the forest yesterday and added to my list I need to buy some milk in the store")
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
        # rs = neo4j_graph_db.create_document_node_cypher(call_of_the_wild_summary, user_id)
        #
        # neo4j_graph_db.query(rs, call_of_the_wild_summary)
        # print(cypher_query)

        # from cognitive_architecture.classifiers.classifier import classify_documents
        #
        # ff = await classify_documents("Lord of the Rings")
        #
        # print(ff)

        # vector_db_namespaces = await get_vectordb_namespace(session, user_id)
        #
        # if vector_db_namespaces == []:
        #     vector_db_namespaces = ["None"]
        #
        #
        # print(vector_db_namespaces)
        # for value in vector_db_namespaces:
        #     print(value)
        #
        #     oo = neo4j_graph_db.update_document_node_with_namespace(user_id,vectordb_namespace= value,document_title="The Call of the Wild")
        #     logging.info("gg", oo)
        #     neo4j_graph_db.query(oo)



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
            "path": ["https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf"],
        }
        await load_documents_to_vectorstore(session, user_id, loader_settings=loader_settings)
        # await user_query_to_graph_db(session, user_id, "I walked in the forest yesterday and added to my list I need to buy some milk in the store")
        # await add_documents_to_graph_db(session, user_id)
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


#1. decompose query
#2. add document to vectordb
#3. add document to graph
#4. fetch relevant memories from semantic, episodic