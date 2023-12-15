
from pydantic import BaseModel, Field
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
from cognitive_architecture.database.postgres.models.user import User
from cognitive_architecture.classifiers.classifier import classify_call
aclient = instructor.patch(OpenAI())
DEFAULT_PRESET = "promethai_chat"
preset_options = [DEFAULT_PRESET]
PROMETHAI_DIR = os.path.join(os.path.expanduser("~"), ".")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
from cognitive_architecture.config import Config
config = Config()
config.load()
from cognitive_architecture.utils import get_document_names
from sqlalchemy.orm import selectinload, joinedload, contains_eager
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from cognitive_architecture.utils import get_document_names, generate_letter_uuid, get_memory_name_by_doc_id, get_unsumarized_vector_db_namespace, get_vectordb_namespace, get_vectordb_document_name

async def fetch_document_vectordb_namespace(session: AsyncSession, user_id: str, namespace_id:str):
    memory = await Memory.create_memory(user_id, session, namespace=namespace_id, memory_label=namespace_id)


    # Managing memory attributes
    existing_user = await Memory.check_existing_user(user_id, session)
    print("here is the existing user", existing_user)
    await memory.manage_memory_attributes(existing_user)
    print("Namespace id is %s", namespace_id)
    await memory.add_dynamic_memory_class(namespace_id.lower(), namespace_id)

    dynamic_memory_class = getattr(memory, namespace_id.lower(), None)

    methods_to_add = ["add_memories", "fetch_memories", "delete_memories"]

    if dynamic_memory_class is not None:
        for method_name in methods_to_add:
            await memory.add_method_to_class(dynamic_memory_class, method_name)
            print(f"Memory method {method_name} has been added")
    else:
        print(f"No attribute named  in memory.")

    print("Available memory classes:", await memory.list_memory_classes())
    result = await memory.dynamic_method_call(dynamic_memory_class, 'fetch_memories',
                                                    observation="placeholder", search_type="summary")

    return result, namespace_id

async def load_documents_to_vectorstore(session: AsyncSession, user_id: str, content:str=None, job_id:str=None, loader_settings:dict=None):
    namespace_id = str(generate_letter_uuid()) + "_" + "SEMANTICMEMORY"
    namespace_class = namespace_id + "_class"

    logging.info("Namespace created with id %s", namespace_id)
    try:
        new_user = User(id=user_id)
        await add_entity(session, new_user)
    except:
        pass

    if job_id is None:
        job_id = str(uuid.uuid4())

    await add_entity(
        session,
        Operation(
            id=job_id,
            user_id=user_id,
            operation_status="RUNNING",
            operation_type="DATA_LOAD",
        ),
    )
    memory = await Memory.create_memory(user_id, session, namespace=namespace_id, job_id=job_id, memory_label=namespace_id)

    if content is not None:
        document_names = [content[:30]]
    if loader_settings is not None:
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
                                                        observation=content, params=params, loader_settings=loader_settings)

        await update_entity(session, Operation, job_id, "SUCCESS")
        return result, namespace_id


async def user_query_to_graph_db(session: AsyncSession, user_id: str, query_input: str):

    try:
        new_user = User(id=user_id)
        await add_entity(session, new_user)
    except:
        pass

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
    cypher_query = await neo4j_graph_db.generate_cypher_query_for_user_prompt_decomposition(user_id,query_input)
    result = neo4j_graph_db.query(cypher_query)

    await update_entity(session, Operation, job_id, "SUCCESS")

    return result




async def add_documents_to_graph_db(session: AsyncSession, user_id: str= None, loader_settings:dict=None, stupid_local_testing_flag=False): #clean this up Vasilije, don't be sloppy
    """"""
    try:
        # await update_document_vectordb_namespace(postgres_session, user_id)
        memory_names, docs = await get_unsumarized_vector_db_namespace(session, user_id)
        logging.info("Memory names are", memory_names)
        logging.info("Docs are", docs)
        for doc, memory_name in zip(docs, memory_names):
            doc_name, doc_id = doc
            if stupid_local_testing_flag:
                classification = [{
                  "DocumentCategory": "Literature",
                  "Title": "Bartleby, the Scrivener",
                  "Summary": "The document is a narrative about an enigmatic copyist named Bartleby who works in a law office. Despite initially being a diligent employee, Bartleby begins to refuse tasks with the phrase 'I would prefer not to' and eventually stops working altogether. His passive resistance and mysterious behavior confound the narrator, who is also his employer. Bartleby's refusal to leave the office leads to various complications, and he is eventually taken to the Tombs as a vagrant. The story ends with Bartleby's death and the revelation that he may have previously worked in the Dead Letter Office, which adds a layer of poignancy to his character.",
                  "d_id": "2a5c571f-bad6-4649-a4ac-36e4bb4f34cd"
                },
                    {
                        "DocumentCategory": "Science",
                        "Title": "The Mysterious World of Quantum Mechanics",
                        "Summary": "This article delves into the fundamentals of quantum mechanics, exploring its paradoxical nature where particles can exist in multiple states simultaneously. It discusses key experiments and theories that have shaped our understanding of the quantum world, such as the double-slit experiment, Schrödinger's cat, and quantum entanglement. The piece also touches upon the implications of quantum mechanics for future technology, including quantum computing and cryptography.",
                        "d_id": "f4e2c3b1-4567-8910-11a2-b3c4d5e6f7g8"
                    },
                    {
                        "DocumentCategory": "History",
                        "Title": "The Rise and Fall of the Roman Empire",
                        "Summary": "This essay provides an overview of the Roman Empire's history, from its foundation to its eventual decline. It examines the political, social, and economic factors that contributed to the empire's expansion and success, as well as those that led to its downfall. Key events and figures such as Julius Caesar, the Punic Wars, and the transition from republic to empire are discussed. The essay concludes with an analysis of the empire's lasting impact on Western civilization.",
                        "d_id": "8h7g6f5e-4d3c-2b1a-09e8-d7c6b5a4f3e2"
                    },
                    {
                        "DocumentCategory": "Technology",
                        "Title": "The Future of Artificial Intelligence",
                        "Summary": "This report explores the current state and future prospects of artificial intelligence (AI). It covers the evolution of AI from simple algorithms to advanced neural networks capable of deep learning. The document discusses various applications of AI in industries such as healthcare, finance, and transportation, as well as ethical considerations and potential risks associated with AI development. Predictions for future advancements and their societal impact are also presented.",
                        "d_id": "3c2b1a09-d8e7-f6g5-h4i3-j1k2l3m4n5o6"
                    },
                    {
                        "DocumentCategory": "Economics",
                        "Title": "Global Economic Trends and Predictions",
                        "Summary": "This analysis examines major trends in the global economy, including the rise of emerging markets, the impact of technology on job markets, and shifts in international trade. It delves into the economic effects of recent global events, such as pandemics and geopolitical conflicts, and discusses how these might shape future economic policies and practices. The document provides predictions for economic growth, inflation rates, and currency fluctuations in the coming years.",
                        "d_id": "7k6j5h4g-3f2e-1d0c-b8a9-m7n6o5p4q3r2"
                    }
                ]
                for classification in classification:

                    neo4j_graph_db = Neo4jGraphDB(url=config.graph_database_url, username=config.graph_database_username,
                                                  password=config.graph_database_password)
                    rs = neo4j_graph_db.create_document_node_cypher(classification, user_id)
                    neo4j_graph_db.query(rs, classification)

                    # select doc from the store
                    neo4j_graph_db.update_document_node_with_namespace(user_id, vectordb_namespace=memory_name, document_id=doc_id)
            else:
                try:
                    classification_content = fetch_document_vectordb_namespace(session, user_id, memory_name)
                except:
                    classification_content = "None"

                classification = await classify_documents(doc_name, document_id =doc_id, content=classification_content)

                logging.info("Classification is", str(classification))
                neo4j_graph_db = Neo4jGraphDB(url=config.graph_database_url, username=config.graph_database_username,
                                              password=config.graph_database_password)
                rs = neo4j_graph_db.create_document_node_cypher(classification, user_id)
                neo4j_graph_db.query(rs, classification)

                # select doc from the store
                neo4j_graph_db.update_document_node_with_namespace(user_id, vectordb_namespace=memory_name,
                                                                   document_id=doc_id)
                await update_entity(session, DocsModel, doc_id, True)
    except:
        pass

class ResponseString(BaseModel):
    response: str = Field(..., default_factory=list)


#

def generate_graph(input) -> ResponseString:
    out =  aclient.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": f"""Use the given context to answer query and use help of associated context: {input}. """,

            },
            {   "role":"system", "content": """You are a top-tier algorithm
                designed for using context summaries based on cognitive psychology to answer user queries, and provide a simple response. 
                Do not mention anything explicit about cognitive architecture, but use the context to answer the query."""}
        ],
        response_model=ResponseString,
    )
    return out
async def user_context_enrichment(session, user_id:str, query:str)->str:
    """
    Asynchronously enriches the user context by integrating various memory systems and document classifications.

    This function uses cognitive architecture to access and manipulate different memory systems (semantic, episodic, and procedural) associated with a user. It fetches memory details from a Neo4j graph database, classifies document categories based on the user's query, and retrieves document IDs for relevant categories. The function also dynamically manages memory attributes and methods, extending the context with document store information to enrich the user's query response.

    Parameters:
    - session (AsyncSession): The database session for executing queries.
    - user_id (str): The unique identifier of the user.
    - query (str): The original query from the user.

    Returns:
    - str: The final enriched context after integrating various memory systems and document classifications.

    The function performs several key operations:
    1. Retrieves semantic and episodic memory details for the user from the Neo4j graph database.
    2. Logs and classifies document categories relevant to the user's query.
    3. Fetches document IDs from Neo4j and corresponding memory names from a PostgreSQL database.
    4. Dynamically manages memory attributes and methods, including the addition of methods like 'add_memories', 'fetch_memories', and 'delete_memories' to the memory class.
    5. Extends the context with document store information relevant to the user's query.
    6. Generates and logs the final result after processing and integrating all information.

    Raises:
    - Exception: Propagates any exceptions that occur during database operations or memory management.

    Example Usage:
    ```python
    enriched_context = await user_context_enrichment(session, "user123", "How does cognitive architecture work?")
    ```
    """
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


    logging.info("Context from graphdb is %s", context)
    document_categories_query = await neo4j_graph_db.get_document_categories(user_id=user_id)
    result = neo4j_graph_db.query(document_categories_query)
    categories = [record["category"] for record in result]
    logging.info('Possible document categories are', str(categories))
    relevant_categories = await classify_call( query= query, context = context, document_types=str(categories))
    logging.info("Relevant categories after the classifier are %s", relevant_categories)

    get_doc_ids = await neo4j_graph_db.get_document_ids(user_id, relevant_categories)

    postgres_id  = neo4j_graph_db.query(get_doc_ids)
    logging.info("Postgres ids are %s", postgres_id)
    namespace_id = await get_memory_name_by_doc_id(session, postgres_id[0]["d_id"])
    logging.info("Namespace ids are %s", namespace_id)
    namespace_id = namespace_id[0]
    namespace_class = namespace_id + "_class"

    memory = await Memory.create_memory(user_id, session, namespace=namespace_id, job_id="23232",
                                        memory_label=namespace_id)

    existing_user = await Memory.check_existing_user(user_id, session)
    print("here is the existing user", existing_user)
    await memory.manage_memory_attributes(existing_user)

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
    result = await memory.dynamic_method_call(dynamic_memory_class, 'fetch_memories',
                                              observation=query)
    context_extension = "Document store information that can help and enrich the anwer is: " + str(result)
    entire_context = context + context_extension
    final_result = generate_graph(entire_context)
    logging.info("Final result is %s", final_result)

    return final_result




async def main():
    user_id = "user"

    async with session_scope(AsyncSessionLocal()) as session:
        # out = await get_vectordb_namespace(session, user_id)
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
            "source": "DEVICE",
            "path": [".data"],
            "strategy": "SUMMARY",
        }
        await load_documents_to_vectorstore(session, user_id, loader_settings=loader_settings)
        await user_query_to_graph_db(session, user_id, "I walked in the forest yesterday and added to my list I need to buy some milk in the store and get a summary from a classical book i read yesterday")
        await add_documents_to_graph_db(session, user_id, loader_settings=loader_settings)
        await user_context_enrichment(session, user_id, query="Tell me about the book I read yesterday")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

