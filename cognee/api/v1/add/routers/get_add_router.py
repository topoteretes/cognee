from uuid import UUID
from fastapi import Form, UploadFile, Depends, Query
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List, Optional, Union, Dict
from pydantic import BaseModel
import subprocess
from cognee.modules.data.methods import get_dataset
from cognee.shared.logging_utils import get_logger
import requests

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = get_logger()


# Pydantic model for text payload
class TextPayload(BaseModel):
    content: str


def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def add(
        data: Union[List[UploadFile], TextPayload],
        datasetId: Optional[UUID] = Query(default=None),
        datasetName: Optional[str] = Query(default=None),
        user: User = Depends(get_authenticated_user),
    ):
        """This endpoint is responsible for adding data to the graph."""
        from cognee.api.v1.add import add as cognee_add

        if not datasetId and not datasetName:
            raise ValueError("Either datasetId or datasetName must be provided.")

        if datasetId and not datasetName:
            dataset = await get_dataset(user_id=user.id, dataset_id=datasetId)
            try:
                datasetName = dataset.name
            except IndexError:
                raise ValueError("No dataset found with the provided datasetName.")

        try:
            if isinstance(data, TextPayload):
                # Handle text payload
                await cognee_add(data.content, datasetName, user=user)
            elif isinstance(data, list) and all(isinstance(item, UploadFile) for item in data):
                # Handle file uploads
                await cognee_add(data, datasetName, user=user)
            elif isinstance(data, str) and data.startswith("http"):
                # This block handles cases where 'data' is a string URL.
                # This implies 'data' can be 'str', which should be in the Union type if this is a primary supported path.
                logger.warning(
                    f"Processing 'data' as a string URL ('{data[:100]}...'). "
                    "This input type should ideally be part of the endpoint's Union type if fully supported."
                )
                if "github" in data:
                    repo_name = data.split("/")[-1].replace(".git", "")
                    subprocess.run(["git", "clone", data, f".data/{repo_name}"], check=True)
                    # Ensure cognee_add is called with consistent parameters
                    await cognee_add(f"data://.data/{repo_name}", datasetName, user=user)
                else:
                    response = requests.get(data)
                    response.raise_for_status()
                    file_data = response.content  # .content is synchronous (bytes)
                    # Ensure cognee_add is called with consistent parameters
                    await cognee_add(file_data, datasetName, user=user)
            else:
                # If data is not TextPayload, List[UploadFile], or a string URL, it's an unhandled type.
                error_message = f"Unsupported data type: {type(data)}. Expected TextPayload, List[UploadFile], or a valid string URL."
                if isinstance(data, str): # If it's a string but not an http URL
                    error_message = f"Received raw string data that is not a valid URL: '{data[:100]}...'"
                logger.error(error_message)
                return JSONResponse(status_code=400, content={"error": error_message})
        except Exception as error:
            logger.error(f"Error processing add request: {error}", exc_info=True)
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
