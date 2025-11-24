from fastapi.responses import JSONResponse
from fastapi import File, UploadFile, Depends, Form
from typing import Optional
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from typing import List
from uuid import UUID
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunErrored,
)

logger = get_logger()


def get_update_router() -> APIRouter:
    router = APIRouter()

    @router.patch("", response_model=None)
    async def update(
        data_id: UUID,
        dataset_id: UUID,
        data: List[UploadFile] = File(default=None),
        node_set: Optional[List[str]] = Form(default=[""], example=[""]),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Update data in a dataset.

        This endpoint updates existing documents in a specified dataset by providing the data_id of the existing document
        to update and the new document with the changes as the data.
        The document is updated, analyzed, and the changes are integrated into the knowledge graph.

        ## Request Parameters
        - **data_id** (UUID): UUID of the document to update in Cognee memory
        - **data** (List[UploadFile]): List of files to upload.
        - **datasetId** (Optional[UUID]): UUID of an already existing dataset
        - **node_set** Optional[list[str]]: List of node identifiers for graph organization and access control.
                 Used for grouping related data points in the knowledge graph.

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
            "Update API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "PATCH /v1/update",
                "dataset_id": str(dataset_id),
                "data_id": str(data_id),
                "node_set": str(node_set),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.update import update as cognee_update

        try:
            update_run = await cognee_update(
                data_id=data_id,
                data=data,
                dataset_id=dataset_id,
                user=user,
                node_set=node_set if node_set else None,
            )

            # If any cognify run errored return JSONResponse with proper error status code
            if any(isinstance(v, PipelineRunErrored) for v in update_run.values()):
                return JSONResponse(status_code=420, content=jsonable_encoder(update_run))
            return update_run

        except Exception as error:
            logger.error(f"Error during deletion by data_id: {str(error)}")
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
