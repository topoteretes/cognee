import json
import logging
import os
from enum import Enum
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cognitive_architecture.database.postgres.database import AsyncSessionLocal
from cognitive_architecture.database.postgres.database_crud import session_scope
from cognitive_architecture.vectorstore_manager import Memory
from dotenv import load_dotenv
from main import add_documents_to_graph_db
from cognitive_architecture.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)

logger = logging.getLogger(__name__)


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
app = FastAPI(debug=True)
#
# from auth.cognito.JWTBearer import JWTBearer
# from auth.auth import jwks
#
# auth = JWTBearer(jwks)

from fastapi import Depends


config = Config()
config.load()

class ImageResponse(BaseModel):
    success: bool
    message: str


@app.get(
    "/",
)
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Hello, World, I am alive!"}


@app.get("/health")
def health_check():
    """
    Health check endpoint that returns the server status.
    """
    return {"status": "OK"}




class Payload(BaseModel):
    payload: Dict[str, Any]

@app.post("/add-memory", response_model=dict)
async def add_memory(
    payload: Payload,
    # files: List[UploadFile] = File(...),
):
    try:
        logging.info(" Adding to Memory ")
        decoded_payload = payload.payload
        async with session_scope(session=AsyncSessionLocal()) as session:
            from main import load_documents_to_vectorstore

            output = await load_documents_to_vectorstore(session, decoded_payload['user_id'], loader_settings=decoded_payload['settings'])
            return JSONResponse(content={"response": output}, status_code=200)

    except Exception as e:
        return JSONResponse(
            content={"response": {"error": str(e)}}, status_code=503
        )

@app.post("/user-query-to-graph")
async def user_query_to_graph(payload: Payload):
    try:
        from main import user_query_to_graph_db
        decoded_payload = payload.payload
        # Execute the query - replace this with the actual execution method
        async with session_scope(session=AsyncSessionLocal()) as session:
            # Assuming you have a method in Neo4jGraphDB to execute the query
            result = await user_query_to_graph_db(decoded_payload['user_id'], decoded_payload['query'], session)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document_to_graph_db")
async def document_to_graph_db(payload: Payload):
    try:
        decoded_payload = payload.payload

        # Execute the query - replace this with the actual execution method
        async with session_scope(session=AsyncSessionLocal()) as session:
            # Assuming you have a method in Neo4jGraphDB to execute the query
            result = await add_documents_to_graph_db(session, decoded_payload['user_id'], decoded_payload['settings'])
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/user-document-vectordb")
# async def generate_document_to_vector_db(payload: Payload, ):
#     try:
#         from database.graph_database.graph import Neo4jGraphDB
#         neo4j_graph_db = Neo4jGraphDB(config.graph_database_url, config.graph_database_username,
#                                       config.graph_database_password)
#         decoded_payload = payload.payload
#
#
#         neo4j_graph_db.update_document_node_with_namespace(decoded_payload['user_id'], document_title="The Call of the Wild")
#
#         # Execute the query - replace this with the actual execution method
#         # async with session_scope(session=AsyncSessionLocal()) as session:
#         #     # Assuming you have a method in Neo4jGraphDB to execute the query
#         #     result = await neo4j_graph_db.query(cypher_query, session)
#         #
#         # return result
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



@app.post("/fetch-memory", response_model=dict)
async def fetch_memory(
    payload: Payload,
    # files: List[UploadFile] = File(...),
):
    try:
        logging.info(" Adding to Memory ")
        decoded_payload = payload.payload
        async with session_scope(session=AsyncSessionLocal()) as session:
            memory = await Memory.create_memory(
                decoded_payload["user_id"], session, "SEMANTICMEMORY", namespace="SEMANTICMEMORY"
            )

            # Adding a memory instance
            await memory.add_memory_instance(decoded_payload["memory_object"])

            # Managing memory attributes
            existing_user = await Memory.check_existing_user(
                decoded_payload["user_id"], session
            )
            await memory.manage_memory_attributes(existing_user)
            await memory.add_dynamic_memory_class(
                decoded_payload["memory_object"],
                decoded_payload["memory_object"].upper(),
            )
            memory_class = decoded_payload["memory_object"] + "_class"
            dynamic_memory_class = getattr(memory, memory_class.lower(), None)

            await memory.add_method_to_class(dynamic_memory_class, "add_memories")
            # await memory.add_method_to_class(memory.semanticmemory_class, 'fetch_memories')
            output = await memory.dynamic_method_call(
                dynamic_memory_class,
                "fetch_memories",
                observation=decoded_payload["observation"],
            )
            return JSONResponse(content={"response": output}, status_code=200)

    except Exception as e:
        return JSONResponse(
            content={"response": {"error": str(e)}}, status_code=503
        )

@app.post("/delete-memory", response_model=dict)
async def delete_memory(
    payload: Payload,
    # files: List[UploadFile] = File(...),
):
    try:
        logging.info(" Adding to Memory ")
        decoded_payload = payload.payload
        async with session_scope(session=AsyncSessionLocal()) as session:
            memory = await Memory.create_memory(
                decoded_payload["user_id"], session, namespace="SEMANTICMEMORY"
            )

            # Adding a memory instance
            await memory.add_memory_instance(decoded_payload["memory_object"])

            # Managing memory attributes
            existing_user = await Memory.check_existing_user(
                decoded_payload["user_id"], session
            )
            await memory.manage_memory_attributes(existing_user)
            await memory.add_dynamic_memory_class(
                decoded_payload["memory_object"],
                decoded_payload["memory_object"].upper(),
            )
            memory_class = decoded_payload["memory_object"] + "_class"
            dynamic_memory_class = getattr(memory, memory_class.lower(), None)

            await memory.add_method_to_class(
                dynamic_memory_class, "delete_memories"
            )
            # await memory.add_method_to_class(memory.semanticmemory_class, 'fetch_memories')
            output = await memory.dynamic_method_call(
                dynamic_memory_class,
                "delete_memories",
                namespace=decoded_payload["memory_object"].upper(),
            )
            return JSONResponse(content={"response": output}, status_code=200)

    except Exception as e:
        return JSONResponse(
            content={"response": {"error": str(e)}}, status_code=503
            )


class TestSetType(Enum):
    SAMPLE = "sample"
    MANUAL = "manual"

def get_test_set(test_set_type, folder_path="example_data", payload=None):
    if test_set_type == TestSetType.SAMPLE:
        file_path = os.path.join(folder_path, "test_set.json")
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                return json.load(file)
    elif test_set_type == TestSetType.MANUAL:
        # Check if the manual test set is provided in the payload
        if payload and "manual_test_set" in payload:
            return payload["manual_test_set"]
        else:
            # Attempt to load the manual test set from a file
            pass

    return None


class MetadataType(Enum):
    SAMPLE = "sample"
    MANUAL = "manual"

def get_metadata(metadata_type, folder_path="example_data", payload=None):
    if metadata_type == MetadataType.SAMPLE:
        file_path = os.path.join(folder_path, "metadata.json")
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                return json.load(file)
    elif metadata_type == MetadataType.MANUAL:
        # Check if the manual metadata is provided in the payload
        if payload and "manual_metadata" in payload:
            return payload["manual_metadata"]
        else:
            pass

    return None





# @app.post("/rag-test/rag_test_run", response_model=dict)
# async def rag_test_run(
#     payload: Payload,
#     background_tasks: BackgroundTasks,
# ):
#     try:
#         logging.info("Starting RAG Test")
#         decoded_payload = payload.payload
#         test_set_type = TestSetType(decoded_payload['test_set'])
#
#         metadata_type = MetadataType(decoded_payload['metadata'])
#
#         metadata = get_metadata(metadata_type, payload=decoded_payload)
#         if metadata is None:
#             return JSONResponse(content={"response": "Invalid metadata value"}, status_code=400)
#
#         test_set = get_test_set(test_set_type, payload=decoded_payload)
#         if test_set is None:
#             return JSONResponse(content={"response": "Invalid test_set value"}, status_code=400)
#
#         async def run_start_test(data, test_set, user_id, params, metadata, retriever_type):
#             result = await start_test(data = data, test_set = test_set, user_id =user_id, params =params, metadata =metadata, retriever_type=retriever_type)
#
#         logging.info("Retriever DATA type", type(decoded_payload['data']))
#
#         background_tasks.add_task(
#             run_start_test,
#             decoded_payload['data'],
#             test_set,
#             decoded_payload['user_id'],
#             decoded_payload['params'],
#             metadata,
#             decoded_payload['retriever_type']
#         )
#
#         logging.info("Retriever type", decoded_payload['retriever_type'])
#         return JSONResponse(content={"response": "Task has been started"}, status_code=200)
#
#     except Exception as e:
#         return JSONResponse(
#
#             content={"response": {"error": str(e)}}, status_code=503
#
#         )


# @app.get("/rag-test/{task_id}")
# async def check_task_status(task_id: int):
#     task_status = task_status_db.get(task_id, "not_found")
#
#     if task_status == "not_found":
#         return {"status": "Task not found"}
#
#     return {"status": task_status}

# @app.get("/available-buffer-actions", response_model=dict)
# async def available_buffer_actions(
#     payload: Payload,
#     # files: List[UploadFile] = File(...),
# ):
#     try:
#         decoded_payload = payload.payload
#
#         Memory_ = Memory(user_id=decoded_payload["user_id"])
#
#         await Memory_.async_init()
#
#         # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
#         output = await Memory_._available_operations()
#         return JSONResponse(content={"response": output}, status_code=200)
#
#     except Exception as e:
#         return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)


# @app.post("/run-buffer", response_model=dict)
# async def run_buffer(
#     payload: Payload,
#     # files: List[UploadFile] = File(...),
# ):
#     try:
#         decoded_payload = payload.payload
#
#         Memory_ = Memory(user_id=decoded_payload["user_id"])
#
#         await Memory_.async_init()
#
#         # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
#         output = await Memory_._run_main_buffer(
#             user_input=decoded_payload["prompt"], params=decoded_payload["params"], attention_modulators=decoded_payload["attention_modulators"]
#         )
#         return JSONResponse(content={"response": output}, status_code=200)
#
#     except Exception as e:
#         return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)
#
#
# @app.post("/buffer/create-context", response_model=dict)
# async def create_context(
#     payload: Payload,
#     # files: List[UploadFile] = File(...),
# ):
#     try:
#         decoded_payload = payload.payload
#
#         Memory_ = Memory(user_id=decoded_payload["user_id"])
#
#         await Memory_.async_init()
#
#         # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
#         output = await Memory_._create_buffer_context(
#             user_input=decoded_payload["prompt"], params=decoded_payload["params"], attention_modulators=decoded_payload["attention_modulators"]
#         )
#         return JSONResponse(content={"response": output}, status_code=200)
#
#     except Exception as e:
#         return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)
#
#
# @app.post("/buffer/get-tasks", response_model=dict)
# async def create_context(
#     payload: Payload,
#     # files: List[UploadFile] = File(...),
# ):
#     try:
#         decoded_payload = payload.payload
#
#         Memory_ = Memory(user_id=decoded_payload["user_id"])
#
#         await Memory_.async_init()
#
#         # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
#         output = await Memory_._get_task_list(
#             user_input=decoded_payload["prompt"], params=decoded_payload["params"], attention_modulators=decoded_payload["attention_modulators"]
#         )
#         return JSONResponse(content={"response": output}, status_code=200)
#
#     except Exception as e:
#         return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)
#
#
# @app.post("/buffer/provide-feedback", response_model=dict)
# async def provide_feedback(
#     payload: Payload,
#     # files: List[UploadFile] = File(...),
# ):
#     try:
#         decoded_payload = payload.payload
#
#         Memory_ = Memory(user_id=decoded_payload["user_id"])
#
#         await Memory_.async_init()
#
#         # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
#         if decoded_payload["total_score"] is None:
#
#             output = await Memory_._provide_feedback(
#                 user_input=decoded_payload["prompt"], params=decoded_payload["params"], attention_modulators=None, total_score=decoded_payload["total_score"]
#             )
#             return JSONResponse(content={"response": output}, status_code=200)
#         else:
#             output = await Memory_._provide_feedback(
#                 user_input=decoded_payload["prompt"], params=decoded_payload["params"], attention_modulators=decoded_payload["attention_modulators"], total_score=None
#             )
#             return JSONResponse(content={"response": output}, status_code=200)
#
#
#     except Exception as e:
#         return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)
def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the API server using uvicorn.

    Parameters:
    host (str): The host for the server.
    port (int): The port for the server.
    """
    try:
        logger.info(f"Starting server at {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.exception(f"Failed to start server: {e}")
        # Here you could add any cleanup code or error recovery code.


if __name__ == "__main__":
    start_api_server()
