from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import Form, File, UploadFile as UF, Depends, status
from typing import List, Optional, Union, Literal, Annotated
from pydantic import WithJsonSchema

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.modules.pipelines.models import PipelineRunErrored
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version
from cognee.api.DTO import ErrorResponse

logger = get_logger()

# NOTE: Needed because of: https://github.com/fastapi/fastapi/discussions/14975
#       Once issue is resolved on Swagger side it can be removed.
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]


def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        response_model=dict,
        responses={
            400: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    @log_usage(function_name="POST /v1/add", log_type="api_endpoint")
    async def add(
        data: List[UploadFile] = File(default=None),
        datasetName: Optional[str] = Form(default=None),
        # Note: Literal is needed for Swagger use
        datasetId: Union[UUID, Literal[""], None] = Form(default=None, examples=[""]),
        node_set: Optional[List[str]] = Form(default=[""], example=[""]),
        run_in_background: Optional[bool] = Form(default=False),
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
        - **run_in_background** (Optional[bool]): Run add pipeline asynchronously (default: False).

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
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="Either datasetId or datasetName must be provided.",
                ).model_dump(),
            )

        try:
            add_run = await cognee_add(
                data,
                datasetName,
                user=user,
                dataset_id=datasetId,
                run_in_background=run_in_background or False,
                node_set=node_set
                if node_set != [""]
                else None,  # Transform default node_set endpoint value to None
            )

            if isinstance(add_run, PipelineRunErrored):
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content=ErrorResponse(
                        error="Pipeline run errored",
                        detail=getattr(add_run, "error", None) or str(add_run),
                    ).model_dump(),
                )
            return add_run.model_dump()
        except Exception as error:
            logger.exception("Add failed")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    detail=str(error),
                ).model_dump(),
            )

    return router
