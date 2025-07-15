import os
from uuid import UUID

from fastapi import Form, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List, Optional
from cognee.shared.logging_utils import get_logger
import requests

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = get_logger()


def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    async def add(
        data: List[UploadFile],
        datasetName: Optional[str] = Form(default=None),
        datasetId: Optional[UUID] = Form(default=None),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Add data to a dataset for processing and knowledge graph construction.

        This endpoint accepts various types of data files and adds them to a specified dataset for processing.
        The data is ingested, analyzed, and integrated into the knowledge graph.

        ## Request Parameters
        - **data** (List[UploadFile]): List of files to upload. Regular file uploads.
        - **datasetName** (Optional[str]): Name of the dataset to add data to
        - **datasetId** (Optional[UUID]): UUID of the dataset to add data to

        Either datasetName or datasetId must be provided.

        ## Response
        Returns information about the add operation containing:
        - Status of the operation
        - Details about the processed data
        - Any relevant metadata from the ingestion process

        ## Error Codes
        - **400 Bad Request**: Neither datasetId nor datasetName provided
        - **409 Conflict**: Error during add operation
        - **403 Forbidden**: User doesn't have permission to add to dataset

        ## Notes
        - To add data to datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL is set to True)

        """
        from cognee.api.v1.add import add as cognee_add

        if not datasetId and not datasetName:
            raise ValueError("Either datasetId or datasetName must be provided.")

        try:
            add_run = await cognee_add(data, datasetName, user=user, dataset_id=datasetId)
            return add_run.model_dump()
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.post("/text", response_model=dict)
    async def add_text(
        text_data: List[str] = Form(description="Plain-text data"),
        datasetName: Optional[str] = Form(default=None),
        datasetId: Optional[UUID] = Form(default=None),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Add text data to a dataset for processing and knowledge graph construction.

        This endpoint accepts only text and adds it to a specified dataset for processing. The text is ingested,
        analyzed, and integrated into the knowledge graph.

        ## Request Parameters
        - **text_data** (List[str]): List of text to process. Can also include:
            HTTP URLs (if ALLOW_HTTP_REQUESTS is enabled)
        - **datasetName** (Optional[str]): Name of the dataset to add data to
        - **datasetId** (Optional[UUID]): UUID of the dataset to add data to

        Either datasetName or datasetId must be provided.

        ## Response
        Returns information about the add operation containing:
        - Status of the operation
        - Details about the processed data
        - Any relevant metadata from the ingestion process

        ## Error Codes
        - **400 Bad Request**: Neither datasetId nor datasetName provided
        - **409 Conflict**: Error during add operation
        - **403 Forbidden**: User doesn't have permission to add to dataset

        ## Notes
        - To add data to datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL is set to True)
        - HTTP URLs are fetched and their content is processed
        - The ALLOW_HTTP_REQUESTS environment variable controls URL processing
        """
        from cognee.api.v1.add import add as cognee_add

        if not datasetId and not datasetName:
            raise ValueError("Either datasetId or datasetName must be provided.")

        try:
            if (
                isinstance(text_data, str)
                and text_data.startswith("http")
                and (os.getenv("ALLOW_HTTP_REQUESTS", "true").lower() == "true")
            ):
                # Fetch and store the data from other types of URL using curl
                response = requests.get(text_data)
                response.raise_for_status()

                file_data = await response.content()
                return await cognee_add(file_data, datasetName, user=user, dataset_id=datasetId)
            else:
                add_run = await cognee_add(text_data, datasetName, user=user, dataset_id=datasetId)
                return add_run.model_dump()
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
