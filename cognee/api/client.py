""" FastAPI server for the Cognee API. """
from datetime import datetime
import os
from uuid import UUID
import aiohttp
import uvicorn
import logging
import sentry_sdk
from typing import List, Union, Optional, Literal
from typing_extensions import Annotated
from fastapi import FastAPI, HTTPException, Form, UploadFile, Query, Depends
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cognee.api.DTO import InDTO, OutDTO
from cognee.api.v1.search import SearchType
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.pipelines.models import PipelineRunStatus


# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)
logger = logging.getLogger(__name__)

if os.getenv("ENV", "prod") == "prod":
    sentry_sdk.init(
        dsn = os.getenv("SENTRY_REPORTING_URL"),
        traces_sample_rate = 1.0,
        profiles_sample_rate = 1.0,
    )

from contextlib import asynccontextmanager

app_environment = os.getenv("ENV", "prod")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # from cognee.modules.data.deletion import prune_system, prune_data
    # await prune_data()
    # await prune_system(metadata = True)
    # if app_environment == "local" or app_environment == "dev":
    from cognee.infrastructure.databases.relational import get_relational_engine
    db_engine = get_relational_engine()
    await db_engine.create_database()

    from cognee.modules.users.methods import get_default_user
    await get_default_user()

    yield

app = FastAPI(debug = app_environment != "prod", lifespan = lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["OPTIONS", "GET", "POST", "DELETE"],
    allow_headers = ["*"],
)

from cognee.api.v1.users.routers import get_auth_router, get_register_router,\
    get_reset_password_router, get_verify_router, get_users_router

from cognee.api.v1.permissions.get_permissions_router import get_permissions_router


from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path == "/api/v1/auth/login":
        return JSONResponse(
            status_code = 400,
            content = {"detail": "LOGIN_BAD_CREDENTIALS"},
        )

    return JSONResponse(
        status_code = 400,
        content = jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )

app.include_router(
    get_auth_router(),
    prefix = "/api/v1/auth",
    tags = ["auth"]
)

app.include_router(
    get_register_router(),
    prefix = "/api/v1/auth",
    tags = ["auth"],
)

app.include_router(
    get_reset_password_router(),
    prefix = "/api/v1/auth",
    tags = ["auth"],
)

app.include_router(
    get_verify_router(),
    prefix = "/api/v1/auth",
    tags = ["auth"],
)

app.include_router(
    get_users_router(),
    prefix = "/api/v1/users",
    tags = ["users"],
)

app.include_router(
    get_permissions_router(),
    prefix = "/api/v1/permissions",
    tags = ["permissions"],
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
    return Response(status_code = 200)


class ErrorResponseDTO(BaseModel):
    message: str


class DatasetDTO(OutDTO):
    id: UUID
    name: str
    created_at: datetime
    updated_at: Optional[datetime]
    owner_id: UUID

@app.get("/api/v1/datasets", response_model = list[DatasetDTO])
async def get_datasets(user: User = Depends(get_authenticated_user)):
    try:
        from cognee.modules.data.methods import get_datasets
        datasets = await get_datasets(user.id)

        return datasets
    except Exception as error:
        logger.error(f"Error retrieving datasets: {str(error)}")
        raise HTTPException(status_code = 500, detail = f"Error retrieving datasets: {str(error)}") from error


@app.delete("/api/v1/datasets/{dataset_id}", response_model = None, responses = { 404: { "model": ErrorResponseDTO }})
async def delete_dataset(dataset_id: str, user: User = Depends(get_authenticated_user)):
    from cognee.modules.data.methods import get_dataset, delete_dataset

    dataset = await get_dataset(user.id, dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Dataset ({dataset_id}) not found."
        )

    await delete_dataset(dataset)


@app.get("/api/v1/datasets/{dataset_id}/graph", response_model = str)
async def get_dataset_graph(dataset_id: str, user: User = Depends(get_authenticated_user)):
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


class DataDTO(OutDTO):
    id: UUID
    name: str
    created_at: datetime
    updated_at: Optional[datetime]
    extension: str
    mime_type: str
    raw_data_location: str

@app.get("/api/v1/datasets/{dataset_id}/data", response_model = list[DataDTO], responses = { 404: { "model": ErrorResponseDTO }})
async def get_dataset_data(dataset_id: str, user: User = Depends(get_authenticated_user)):
    from cognee.modules.data.methods import get_dataset_data, get_dataset

    dataset = await get_dataset(user.id, dataset_id)

    if dataset is None:
        return JSONResponse(
            status_code = 404,
            content = ErrorResponseDTO(f"Dataset ({dataset_id}) not found."),
        )

    dataset_data = await get_dataset_data(dataset_id = dataset.id)

    if dataset_data is None:
        return []

    return dataset_data


@app.get("/api/v1/datasets/status", response_model = dict[str, PipelineRunStatus])
async def get_dataset_status(datasets: Annotated[List[str], Query(alias="dataset")] = None, user: User = Depends(get_authenticated_user)):
    from cognee.api.v1.datasets.datasets import datasets as cognee_datasets

    try:
        datasets_statuses = await cognee_datasets.get_status(datasets)

        return datasets_statuses
    except Exception as error:
        return JSONResponse(
            status_code = 409,
            content = {"error": str(error)}
        )


@app.get("/api/v1/datasets/{dataset_id}/data/{data_id}/raw", response_class = FileResponse)
async def get_raw_data(dataset_id: str, data_id: str, user: User = Depends(get_authenticated_user)):
    from cognee.modules.data.methods import get_dataset, get_dataset_data

    dataset = await get_dataset(user.id, dataset_id)

    if dataset is None:
        return JSONResponse(
            status_code = 404,
            content = {
                "detail": f"Dataset ({dataset_id}) not found."
            }
        )

    dataset_data = await get_dataset_data(dataset.id)

    if dataset_data is None:
        raise HTTPException(status_code = 404, detail = f"Dataset ({dataset_id}) not found.")

    data = [data for data in dataset_data if str(data.id) == data_id][0]

    if data is None:
        return JSONResponse(
            status_code = 404,
            content = {
                "detail": f"Data ({data_id}) not found in dataset ({dataset_id})."
            }
        )

    return data.raw_data_location


@app.post("/api/v1/add", response_model = None)
async def add(
    data: List[UploadFile],
    datasetId: str = Form(...),
    user: User = Depends(get_authenticated_user),
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
                user = user,
            )
    except Exception as error:
        return JSONResponse(
            status_code = 409,
            content = {"error": str(error)}
        )


class CognifyPayloadDTO(BaseModel):
    datasets: List[str]

@app.post("/api/v1/cognify", response_model = None)
async def cognify(payload: CognifyPayloadDTO, user: User = Depends(get_authenticated_user)):
    """ This endpoint is responsible for the cognitive processing of the content."""
    from cognee.api.v1.cognify.cognify_v2 import cognify as cognee_cognify
    try:
        await cognee_cognify(payload.datasets, user)
    except Exception as error:
        return JSONResponse(
            status_code = 409,
            content = {"error": str(error)}
        )


class SearchPayloadDTO(InDTO):
    search_type: SearchType
    query: str

@app.post("/api/v1/search", response_model = list)
async def search(payload: SearchPayloadDTO, user: User = Depends(get_authenticated_user)):
    """ This endpoint is responsible for searching for nodes in the graph."""
    from cognee.api.v1.search import search as cognee_search

    try:
        results = await cognee_search(payload.search_type, payload.query, user)

        return results
    except Exception as error:
        return JSONResponse(
            status_code = 409,
            content = {"error": str(error)}
        )

from cognee.modules.settings.get_settings import LLMConfig, VectorDBConfig

class LLMConfigDTO(OutDTO, LLMConfig):
    pass

class VectorDBConfigDTO(OutDTO, VectorDBConfig):
    pass

class SettingsDTO(OutDTO):
    llm: LLMConfigDTO
    vector_db: VectorDBConfigDTO

@app.get("/api/v1/settings", response_model = SettingsDTO)
async def get_settings(user: User = Depends(get_authenticated_user)):
    from cognee.modules.settings import get_settings as get_cognee_settings
    return get_cognee_settings()


class LLMConfigDTO(InDTO):
    provider: Union[Literal["openai"], Literal["ollama"], Literal["anthropic"]]
    model: str
    api_key: str

class VectorDBConfigDTO(InDTO):
    provider: Union[Literal["lancedb"], Literal["qdrant"], Literal["weaviate"]]
    url: str
    api_key: str

class SettingsPayloadDTO(InDTO):
    llm: Optional[LLMConfigDTO] = None
    vector_db: Optional[VectorDBConfigDTO] = None

@app.post("/api/v1/settings", response_model = None)
async def save_settings(new_settings: SettingsPayloadDTO, user: User = Depends(get_authenticated_user)):
    from cognee.modules.settings import save_llm_config, save_vector_db_config

    if new_settings.llm is not None:
        await save_llm_config(new_settings.llm)

    if new_settings.vector_db is not None:
        await save_vector_db_config(new_settings.vector_db)


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the API server using uvicorn.
    Parameters:
    host (str): The host for the server.
    port (int): The port for the server.
    """
    try:
        logger.info("Starting server at %s:%s", host, port)

        uvicorn.run(app, host = host, port = port)
    except Exception as e:
        logger.exception(f"Failed to start server: {e}")
        # Here you could add any cleanup code or error recovery code.


if __name__ == "__main__":
    start_api_server()
