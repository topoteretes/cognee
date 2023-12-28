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
from main import add_documents_to_graph_db, user_context_enrichment
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

            if 'settings' in decoded_payload and decoded_payload['settings'] is not None:
                settings_for_loader = decoded_payload['settings']
            else:
                settings_for_loader = None

            if 'content' in decoded_payload and decoded_payload['content'] is not None:
                content = decoded_payload['content']
            else:
                content = None

            output = await load_documents_to_vectorstore(session, decoded_payload['user_id'], content=content, loader_settings=settings_for_loader)
            return JSONResponse(content={"response": output}, status_code=200)

    except Exception as e:
        return JSONResponse(
            content={"response": {"error": str(e)}}, status_code=503
        )

@app.post("/add-architecture-public-memory", response_model=dict)
async def add_memory(
    payload: Payload,
    # files: List[UploadFile] = File(...),
):
    try:
        logging.info(" Adding to Memory ")
        decoded_payload = payload.payload
        async with session_scope(session=AsyncSessionLocal()) as session:
            from main import load_documents_to_vectorstore
            if 'content' in decoded_payload and decoded_payload['content'] is not None:
                content = decoded_payload['content']
            else:
                content = None

            user_id = 'system_user'
            loader_settings = {
                "format": "PDF",
                "source": "DEVICE",
                "path": [".data"]
            }

            output = await load_documents_to_vectorstore(session, user_id=user_id, content=content, loader_settings=loader_settings)
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
            result = await user_query_to_graph_db(session= session, user_id= decoded_payload['user_id'],query_input =decoded_payload['query'])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document-to-graph-db")
async def document_to_graph_db(payload: Payload):
    logging.info("Adding documents to graph db")
    try:
        decoded_payload = payload.payload
        if 'settings' in decoded_payload and decoded_payload['settings'] is not None:
            settings_for_loader = decoded_payload['settings']
        else:
            settings_for_loader = None
        if 'memory_type' in decoded_payload and decoded_payload['memory_type'] is not None:
            memory_type = decoded_payload['memory_type']
        else:
            memory_type = None
        async with session_scope(session=AsyncSessionLocal()) as session:
            result = await add_documents_to_graph_db(session =session, user_id = decoded_payload['user_id'], document_memory_types =memory_type)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cognitive-context-enrichment")
async def cognitive_context_enrichment(payload: Payload):
    try:
        decoded_payload = payload.payload

        # Execute the query - replace this with the actual execution method
        async with session_scope(session=AsyncSessionLocal()) as session:
            # Assuming you have a method in Neo4jGraphDB to execute the query
            result = await user_context_enrichment(session, user_id = decoded_payload['user_id'], query= decoded_payload['query'], generative_response=decoded_payload['generative_response'], memory_type= decoded_payload['memory_type'])
        return JSONResponse(content={"response": result}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user-query-classifier")
async def user_query_classfier(payload: Payload):
    try:
        decoded_payload = payload.payload

        # Execute the query - replace this with the actual execution method
        async with session_scope(session=AsyncSessionLocal()) as session:
            from cognitive_architecture.classifiers.classifier import classify_user_query
            # Assuming you have a method in Neo4jGraphDB to execute the query
            result = await classify_user_query(session, decoded_payload['user_id'], decoded_payload['query'])
        return JSONResponse(content={"response": result}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/drop-db")
async def drop_db(payload: Payload):
    try:
        decoded_payload = payload.payload

        if decoded_payload['operation'] == 'drop':

            if  os.environ.get('AWS_ENV') == 'dev':
                host = os.environ.get('POSTGRES_HOST')
                username = os.environ.get('POSTGRES_USER')
                password = os.environ.get('POSTGRES_PASSWORD')
                database_name = os.environ.get('POSTGRES_DB')
            else:
                pass

            from cognitive_architecture.database.create_database_tst import drop_database, create_admin_engine

            engine = create_admin_engine(username, password, host, database_name)
            drop_database(engine)
            return JSONResponse(content={"response": "DB dropped"}, status_code=200)
        else:

            if os.environ.get('AWS_ENV') == 'dev':
                host = os.environ.get('POSTGRES_HOST')
                username = os.environ.get('POSTGRES_USER')
                password = os.environ.get('POSTGRES_PASSWORD')
                database_name = os.environ.get('POSTGRES_DB')
            else:
                pass

            from cognitive_architecture.database.create_database_tst import create_database, create_admin_engine

            engine = create_admin_engine(username, password, host, database_name)
            create_database(engine)
            return JSONResponse(content={"response": " DB created"}, status_code=200)



    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


@app.post("/create-public-memory")
async def create_public_memory(payload: Payload):
    try:
        decoded_payload = payload.payload

        if 'user_id' in decoded_payload and decoded_payload['user_id'] is not None:
            user_id = decoded_payload['user_id']
        else:
            user_id = None

        if 'labels' in decoded_payload and decoded_payload['labels'] is not None:
            labels = decoded_payload['labels']
        else:
            labels = None

        if 'topic' in decoded_payload and decoded_payload['topic'] is not None:
            topic = decoded_payload['topic']
        else:
            topic = None

        # Execute the query - replace this with the actual execution method
        # async with session_scope(session=AsyncSessionLocal()) as session:
        #     from main import create_public_memory
            # Assuming you have a method in Neo4jGraphDB to execute the query
        result = await create_public_memory(user_id=user_id, labels=labels, topic=topic)
        return JSONResponse(content={"response": result}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/attach-user-to-public-memory")
async def attach_user_to_public_memory(payload: Payload):
    try:
        decoded_payload = payload.payload

        if 'topic' in decoded_payload and decoded_payload['topic'] is not None:
            topic = decoded_payload['topic']
        else:
            topic = None
        if 'labels' in decoded_payload and decoded_payload['labels'] is not None:
            labels = decoded_payload['labels']
        else:
            labels = ['sr']

        # Execute the query - replace this with the actual execution method
        async with session_scope(session=AsyncSessionLocal()) as session:
            from main import attach_user_to_memory, create_public_memory
            # Assuming you have a method in Neo4jGraphDB to execute the query
            await create_public_memory(user_id=decoded_payload['user_id'], topic=topic, labels=labels)
            result = await attach_user_to_memory( user_id = decoded_payload['user_id'], topic=topic, labels=labels)
        return JSONResponse(content={"response": result}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unlink-user-from-public-memory")
async def unlink_user_from_public_memory(payload: Payload):
    try:
        decoded_payload = payload.payload

        if 'topic' in decoded_payload and decoded_payload['topic'] is not None:
            topic = decoded_payload['topic']
        else:
            topic = None

        # Execute the query - replace this with the actual execution method
        async with session_scope(session=AsyncSessionLocal()) as session:
            from main import unlink_user_from_memory
            # Assuming you have a method in Neo4jGraphDB to execute the query
            result = await unlink_user_from_memory( user_id = decoded_payload['user_id'], topic=topic)
        return JSONResponse(content={"response": result}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
