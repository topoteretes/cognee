import logging
import os
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vectorstore_manager import Memory
from dotenv import load_dotenv
# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)

logger = logging.getLogger(__name__)



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
app = FastAPI(debug=True)

from auth.cognito.JWTBearer import JWTBearer
from auth.auth import jwks
auth = JWTBearer(jwks)

from fastapi import Depends
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

def memory_factory(memory_type):
    load_dotenv()

    class Payload(BaseModel):
        payload: Dict[str, Any]

    @app.post("/{memory_type}/add-memory", response_model=dict)
    async def add_memory(
        payload: Payload,
        # files: List[UploadFile] = File(...),
    ):
        try:
            logging.info(" Init PDF processing")

            decoded_payload = payload.payload

            Memory_ = Memory(user_id=decoded_payload["user_id"])

            await Memory_.async_init()

            memory_class = getattr(Memory_, f"_add_{memory_type}_memory", None)
            output = await memory_class(
                observation=decoded_payload["prompt"],
                loader_settings=decoded_payload["loader_settings"],
                params=decoded_payload["params"],
            )
            return JSONResponse(content={"response": output}, status_code=200)

        except Exception as e:
            return JSONResponse(
                content={"response": {"error": str(e)}}, status_code=503
            )

    @app.post("/{memory_type}/fetch-memory", response_model=dict)
    async def fetch_memory(
        payload: Payload,
        # files: List[UploadFile] = File(...),
    ):
        try:
            decoded_payload = payload.payload

            Memory_ = Memory(user_id=decoded_payload["user_id"])

            await Memory_.async_init()

            memory_class = getattr(Memory_, f"_fetch_{memory_type}_memory", None)
            output = memory_class(observation=decoded_payload["prompt"])
            return JSONResponse(content={"response": output}, status_code=200)

        except Exception as e:
            return JSONResponse(
                content={"response": {"error": str(e)}}, status_code=503
            )

    @app.post("/{memory_type}/delete-memory", response_model=dict)
    async def delete_memory(
        payload: Payload,
        # files: List[UploadFile] = File(...),
    ):
        try:
            decoded_payload = payload.payload

            Memory_ = Memory(user_id=decoded_payload["user_id"])

            await Memory_.async_init()

            memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
            output = memory_class(observation=decoded_payload["prompt"])
            return JSONResponse(content={"response": output}, status_code=200)

        except Exception as e:
            return JSONResponse(
                content={"response": {"error": str(e)}}, status_code=503
            )


memory_list = ["episodic", "buffer", "semantic"]
for memory_type in memory_list:
    memory_factory(memory_type)


@app.get("/available-buffer-actions", response_model=dict)
async def available_buffer_actions(
    payload: Payload,
    # files: List[UploadFile] = File(...),
):
    try:
        decoded_payload = payload.payload

        Memory_ = Memory(user_id=decoded_payload["user_id"])

        await Memory_.async_init()

        # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
        output = await Memory_._available_operations()
        return JSONResponse(content={"response": output}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)


@app.post("/run-buffer", response_model=dict)
async def run_buffer(
    payload: Payload,
    # files: List[UploadFile] = File(...),
):
    try:
        decoded_payload = payload.payload

        Memory_ = Memory(user_id=decoded_payload["user_id"])

        await Memory_.async_init()

        # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
        output = await Memory_._run_main_buffer(
            user_input=decoded_payload["prompt"], params=decoded_payload["params"], attention_modulators=decoded_payload["attention_modulators"]
        )
        return JSONResponse(content={"response": output}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)


@app.post("/buffer/create-context", response_model=dict)
async def create_context(
    payload: Payload,
    # files: List[UploadFile] = File(...),
):
    try:
        decoded_payload = payload.payload

        Memory_ = Memory(user_id=decoded_payload["user_id"])

        await Memory_.async_init()

        # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
        output = await Memory_._create_buffer_context(
            user_input=decoded_payload["prompt"], params=decoded_payload["params"], attention_modulators=decoded_payload["attention_modulators"]
        )
        return JSONResponse(content={"response": output}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)


@app.post("/buffer/get-tasks", response_model=dict)
async def create_context(
    payload: Payload,
    # files: List[UploadFile] = File(...),
):
    try:
        decoded_payload = payload.payload

        Memory_ = Memory(user_id=decoded_payload["user_id"])

        await Memory_.async_init()

        # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
        output = await Memory_._get_task_list(
            user_input=decoded_payload["prompt"], params=decoded_payload["params"], attention_modulators=decoded_payload["attention_modulators"]
        )
        return JSONResponse(content={"response": output}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)


@app.post("/buffer/provide-feedback", response_model=dict)
async def provide_feedback(
    payload: Payload,
    # files: List[UploadFile] = File(...),
):
    try:
        decoded_payload = payload.payload

        Memory_ = Memory(user_id=decoded_payload["user_id"])

        await Memory_.async_init()

        # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
        if decoded_payload["total_score"] is None:

            output = await Memory_._provide_feedback(
                user_input=decoded_payload["prompt"], params=decoded_payload["params"], attention_modulators=None, total_score=decoded_payload["total_score"]
            )
            return JSONResponse(content={"response": output}, status_code=200)
        else:
            output = await Memory_._provide_feedback(
                user_input=decoded_payload["prompt"], params=decoded_payload["params"], attention_modulators=decoded_payload["attention_modulators"], total_score=None
            )
            return JSONResponse(content={"response": output}, status_code=200)


    except Exception as e:
        return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)
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
