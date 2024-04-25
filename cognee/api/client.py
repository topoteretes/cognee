""" FastAPI server for the Cognee API. """
import os
from uuid import UUID

import aiohttp
import uvicorn
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)

logger = logging.getLogger(__name__)

from cognee.config import Config

config = Config()
config.load()

from typing import Dict, Any, List, Union, BinaryIO
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(debug=True)
#
# from auth.cognito.JWTBearer import JWTBearer
# from auth.auth import jwks
#
# auth = JWTBearer(jwks)


@app.get("/")
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


class AddPayload(BaseModel):
    data: Union[str, BinaryIO, List[Union[str, BinaryIO]]]
    dataset_id: UUID
    dataset_name: str
    class Config:
        arbitrary_types_allowed = True # This is required to allow the use of Union
class CognifyPayload(BaseModel):
    datasets: Union[str, List[str]]

class SearchPayload(BaseModel):
    query_params:  Dict[str, Any]
@app.post("/add", response_model=dict)
async def add(payload: AddPayload):
    """ This endpoint is responsible for adding data to the graph."""
    from v1.add.add import add

    try:
        data = payload.data
        if isinstance(data, str) and data.startswith('http'):
            if 'github' in data:
                # Perform git clone if the URL is from GitHub
                repo_name = data.split('/')[-1].replace('.git', '')
                os.system(f'git clone {data} .data/{repo_name}')
                await add(
                    "data://.data/",
                    f"{repo_name}",
                )
            else:
                # Fetch and store the data from other types of URL using curl
                async with aiohttp.ClientSession() as session:
                    async with session.get(data) as resp:
                        if resp.status == 200:
                            file_data = await resp.read()
                            with open(f'.data/{data.split("/")[-1]}', 'wb') as f:
                                f.write(file_data)
                            await add(
                                f"data://.data/",
                                f"{data.split('/')[-1]}",
                            )
        else:
            await add(
                payload.data,
                payload.dataset_name,
            )
    except Exception as error:
        return JSONResponse(
            status_code = 409,
            content = { "error": str(error) }
        )

@app.post("/cognify", response_model=dict)
async def cognify(payload: CognifyPayload):
    """ This endpoint is responsible for the cognitive processing of the content."""
    from v1.cognify.cognify import cognify

    try:
        await cognify(payload.datasets)
    except Exception as error:
        return JSONResponse(
            status_code = 409,
            content = { "error": error }
        )


@app.post("/search", response_model=dict)
async def search(payload: SearchPayload):
    """ This endpoint is responsible for searching for nodes in the graph."""
    from v1.search.search import search

    try:
        search_type = 'SIMILARITY'
        await search(search_type, payload.query_params)
    except Exception as error:
        return JSONResponse(
            status_code = 409,
            content = { "error": error }
        )



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
