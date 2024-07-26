""" FastAPI server for the Cognee API. """
import os
import aiohttp
import uvicorn
import json
import asyncio
import logging
import sentry_sdk
from typing import Dict, Any, List, Union, Optional, Literal
from typing_extensions import Annotated
from fastapi import FastAPI, HTTPException, Form, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)
logger = logging.getLogger(__name__)

if os.getenv("ENV") == "prod":
    sentry_sdk.init(
        dsn = os.getenv("SENTRY_REPORTING_URL"),
        traces_sample_rate = 1.0,
        profiles_sample_rate = 1.0,
    )

app = FastAPI(debug = os.getenv("ENV") != "prod")

origins = [
    "http://frontend:3000",
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["OPTIONS", "GET", "POST", "DELETE"],
    allow_headers=["*"],
)

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

@app.get("/datasets", response_model=list)
async def get_datasets():
    from cognee.api.v1.datasets.datasets import datasets
    return datasets.list_datasets()

@app.delete("/datasets/{dataset_id}", response_model=dict)
async def delete_dataset(dataset_id: str):
    from cognee.api.v1.datasets.datasets import datasets
    datasets.delete_dataset(dataset_id)
    return JSONResponse(
        status_code=200,
        content="OK",
    )

@app.get("/datasets/{dataset_id}/graph", response_model=list)
async def get_dataset_graph(dataset_id: str):
    from cognee.shared.utils import render_graph
    from cognee.infrastructure.databases.graph import get_graph_engine

    try:
        graph_client = await get_graph_engine()
        graph_url = await render_graph(graph_client.graph)

        return JSONResponse(
            status_code = 200,
            content = str(graph_url),
        )
    except:
        return JSONResponse(
            status_code = 409,
            content = "Graphistry credentials are not set. Please set them in your .env file.",
        )

@app.get("/datasets/{dataset_id}/data", response_model=list)
async def get_dataset_data(dataset_id: str):
    from cognee.api.v1.datasets.datasets import datasets
    dataset_data = datasets.list_data(dataset_id)
    if dataset_data is None:
        raise HTTPException(status_code=404, detail=f"Dataset ({dataset_id}) not found.")
    return [
        dict(
            id=data["id"],
            name=f"{data['name']}.{data['extension']}",
            filePath=data["file_path"],
            mimeType=data["mime_type"],
        )
        for data in dataset_data
    ]

@app.get("/datasets/status", response_model=dict)
async def get_dataset_status(datasets: Annotated[List[str], Query(alias="dataset")] = None):
    from cognee.api.v1.datasets.datasets import datasets as cognee_datasets
    datasets_statuses = cognee_datasets.get_status(datasets)

    return JSONResponse(
        status_code = 200,
        content = datasets_statuses,
    )

@app.get("/datasets/{dataset_id}/data/{data_id}/raw", response_class=FileResponse)
async def get_raw_data(dataset_id: str, data_id: str):
    from cognee.api.v1.datasets.datasets import datasets
    dataset_data = datasets.list_data(dataset_id)
    if dataset_data is None:
        raise HTTPException(status_code=404, detail=f"Dataset ({dataset_id}) not found.")
    data = [data for data in dataset_data if data["id"] == data_id][0]
    return data["file_path"]

class AddPayload(BaseModel):
    data: Union[str, UploadFile, List[Union[str, UploadFile]]]
    dataset_id: str
    class Config:
        arbitrary_types_allowed = True

@app.post("/add", response_model=dict)
async def add(
    data: List[UploadFile],
    datasetId: str = Form(...),
):
    """ This endpoint is responsible for adding data to the graph."""
    from cognee.api.v1.add import add as cognee_add
    try:
        if isinstance(data, str) and data.startswith("http"):
            if "github" in data:
                # Perform git clone if the URL is from GitHub
                repo_name = data.split("/")[-1].replace(".git", "")
                os.system(f"git clone {data} .data/{repo_name}")
                await cognee_add(
                    "data://.data/",
                    f"{repo_name}",
                )
            else:
                # Fetch and store the data from other types of URL using curl
                async with aiohttp.ClientSession() as session:
                    async with session.get(data) as resp:
                        if resp.status == 200:
                            file_data = await resp.read()
                            with open(f".data/{data.split('/')[-1]}", "wb") as f:
                                f.write(file_data)
                            await cognee_add(
                                "data://.data/",
                                f"{data.split('/')[-1]}",
                            )
        else:
            await cognee_add(
                data,
                datasetId,
            )
            return JSONResponse(
                status_code=200,
                content="OK"
            )
    except Exception as error:
        return JSONResponse(
            status_code=409,
            content={"error": str(error)}
        )

class CognifyPayload(BaseModel):
    datasets: List[str]

@app.post("/cognify", response_model=dict)
async def cognify(payload: CognifyPayload):
    """ This endpoint is responsible for the cognitive processing of the content."""
    from cognee.api.v1.cognify.cognify_v2 import cognify as cognee_cognify
    try:
        await cognee_cognify(payload.datasets)
        return JSONResponse(
            status_code = 200,
            content = "OK"
        )
    except Exception as error:
        return JSONResponse(
            status_code = 409,
            content = {"error": str(error)}
        )

class SearchPayload(BaseModel):
    query_params: Dict[str, Any]

@app.post("/search", response_model=dict)
async def search(payload: SearchPayload):
    """ This endpoint is responsible for searching for nodes in the graph."""
    from cognee.api.v1.search import search as cognee_search
    try:
        search_type = payload.query_params["searchType"]
        params = {
            "query": payload.query_params["query"],
        }
        results = await cognee_search(search_type, params)
        return JSONResponse(
            status_code=200,
            content=json.dumps(results)
        )
    except Exception as error:
        return JSONResponse(
            status_code=409,
            content={"error": str(error)}
        )

@app.get("/settings", response_model=dict)
async def get_settings():
    from cognee.modules.settings import get_settings
    return get_settings()

class LLMConfig(BaseModel):
    provider: Union[Literal["openai"], Literal["ollama"], Literal["anthropic"]]
    model: str
    apiKey: str

class VectorDBConfig(BaseModel):
    provider: Union[Literal["lancedb"], Literal["qdrant"], Literal["weaviate"]]
    url: str
    apiKey: str

class SettingsPayload(BaseModel):
    llm: Optional[LLMConfig] = None
    vectorDB: Optional[VectorDBConfig] = None

@app.post("/settings", response_model=dict)
async def save_config(new_settings: SettingsPayload):
    from cognee.modules.settings import save_llm_config, save_vector_db_config
    if new_settings.llm is not None:
        await save_llm_config(new_settings.llm)
    if new_settings.vectorDB is not None:
        await save_vector_db_config(new_settings.vectorDB)
    return JSONResponse(
        status_code=200,
        content="OK",
    )


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the API server using uvicorn.
    Parameters:
    host (str): The host for the server.
    port (int): The port for the server.
    """
    try:
        logger.info("Starting server at %s:%s", host, port)

        from cognee.infrastructure.databases.relational import get_relationaldb_config
        relational_config = get_relationaldb_config()
        relational_config.create_engine()

        from cognee.modules.data.deletion import prune_system, prune_data
        asyncio.run(prune_data())
        asyncio.run(prune_system(metadata = True))

        uvicorn.run(app, host = host, port = port)
    except Exception as e:
        logger.exception(f"Failed to start server: {e}")
        # Here you could add any cleanup code or error recovery code.


if __name__ == "__main__":
    start_api_server()
