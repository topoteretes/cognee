from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import Form, File, UploadFile, Depends
from typing import List, Optional, Union, Literal

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.modules.pipelines.models import PipelineRunErrored
from cognee.shared.logging_utils import get_logger
from cognee import __version__ as cognee_version

logger = get_logger()


def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    async def add(
        data: List[UploadFile] = File(default=None),
        datasetName: Optional[str] = Form(default=None),
        # Note: Literal is needed for Swagger use
        datasetId: Union[UUID, Literal[""], None] = Form(default=None, examples=[""]),
        node_set: Optional[List[str]] = Form(default=[""], example=[""]),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Add data to a dataset for processing and knowledge graph construction.

        This endpoint accepts various types of data (files, URLs, GitHub repositories)
        and adds them to a specified dataset for processing. The data is ingested,
        analyzed, and integrated into the knowledge graph.

        ## Request Parameters
        - **data** (List[UploadFile]): List of files to upload. Can also include:
          - HTTP URLs (if ALLOW_HTTP_REQUESTS is enabled)
          - GitHub repository URLs (will be cloned and processed)
          - Regular file uploads
        - **datasetName** (Optional[str]): Name of the dataset to add data to
        - **datasetId** (Optional[UUID]): UUID of an already existing dataset
        - **node_set** Optional[list[str]]: List of node identifiers for graph organization and access control.
                 Used for grouping related data points in the knowledge graph.

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
        - datasetId value can only be the UUID of an already existing dataset
        """
        send_telemetry(
            "Add API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/add",
                "node_set": node_set,
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.add import add as cognee_add

        if not datasetId and not datasetName:
            raise ValueError("Either datasetId or datasetName must be provided.")

        try:
            add_run = await cognee_add(
                data,
                datasetName,
                user=user,
                dataset_id=datasetId,
                node_set=node_set if node_set else None,
            )

            if isinstance(add_run, PipelineRunErrored):
                return JSONResponse(status_code=420, content=add_run.model_dump(mode="json"))
            return add_run.model_dump()
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
