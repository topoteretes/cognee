from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import Form, File, Depends, status
from typing import List, Optional, Union, Literal, Annotated
from pydantic import WithJsonSchema

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.tasks.ingestion.data_item import (
    pair_labels_with_data,
    parse_external_metadata,
    parse_labels,
)
from cognee.shared.utils import send_telemetry
from cognee.modules.pipelines.models import PipelineRunErrored
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunInfo
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version
from cognee.api.DTO import ErrorResponse
from cognee.api.upload_fields import OptionalUploadFile, drop_blank_uploads

logger = get_logger()

# Swagger UI prefills newly added array items from the ITEM-level example;
# without one it inserts the literal "string". An empty item example keeps
# "Add item" runnable (empty entries are filtered out server-side).
EmptyExampleStr = Annotated[str, WithJsonSchema({"type": "string", "example": ""})]


def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        response_model=PipelineRunInfo,
        responses={
            400: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    @log_usage(function_name="POST /v1/add", log_type="api_endpoint")
    async def add(
        data: List[OptionalUploadFile] = File(default=None),
        raw_data: Optional[List[EmptyExampleStr]] = Form(
            default=None,
            examples=[[]],
            description=(
                "Data given as strings instead of uploads, one entry each: raw text to "
                "ingest, a local file or directory path on the server's filesystem "
                "(requires ACCEPT_LOCAL_FILE_PATH), a web URL to fetch (requires "
                "ALLOW_HTTP_REQUESTS), or a GitHub/GitLab repository URL, which is "
                "shallow-cloned and indexed as a code graph by cognify. Combined with "
                "'data': uploads come first, then these entries; labels and "
                "external_metadata pair with that combined order. Empty entries are "
                "ignored."
            ),
        ),
        labels: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                'Per-item labels, e.g. ["finance", "people", ""] — the Nth label applies '
                "to the Nth data item (uploads first, then raw_data entries), one entry "
                "per item, an empty entry skips that item. The comma-separated form "
                '"finance,people," is accepted equivalently (it is what Swagger UI sends '
                "when you type a JSON array here), so labels cannot contain commas unless "
                "the client sends real JSON. Stored on each item's data record and "
                "returned when listing dataset data."
            ),
        ),
        external_metadata: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                "JSON array of per-file metadata objects, e.g. "
                '[{"source": "crm", "ticket": 42}, null]. Paired positionally like labels: '
                "the Nth entry applies to the Nth uploaded file (null or {} skips that "
                "file), and one entry per file is required when any is given. Merged into "
                "the file's stored external_metadata (your keys win over loader-derived "
                "ones; 'node_set' is reserved) and returned when listing dataset data."
            ),
        ),
        datasetName: Optional[str] = Form(
            default=None,
            examples=["default_dataset"],
            description=(
                "Name of the target dataset (created if it does not exist). "
                "Required unless datasetId is provided."
            ),
        ),
        # Note: Literal is needed for Swagger use
        datasetId: Union[UUID, Literal[""], None] = Form(
            default=None,
            examples=[""],
            description=(
                "Providing dataset ID is mandatory for sharing a dataset between users. Datasets provided by name will only be resolvable by dataset owner."
            ),
        ),
        node_set: Optional[List[str]] = Form(default=[""], example=[""]),
        run_in_background: Optional[bool] = Form(default=False),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Add data to a dataset for processing and knowledge graph construction.

        This endpoint accepts file uploads and string inputs (text, server-side file
        paths, web URLs, GitHub/GitLab repository URLs) and adds them to a specified
        dataset for processing. The data is ingested, analyzed, and integrated into
        the knowledge graph.

        ## Request Parameters
        - **data** (List[UploadFile]): Files to upload.
        - **raw_data** (Optional[List[str]]): String inputs, one entry each:
          - Raw text to ingest
          - A local file or directory path on the server (requires ACCEPT_LOCAL_FILE_PATH)
          - A web URL, fetched as a page (requires ALLOW_HTTP_REQUESTS)
          - A GitHub/GitLab repository URL, shallow-cloned and indexed as a code graph
          At least one of data or raw_data is required. Uploads come first, then
          raw_data entries; labels and external_metadata pair with that combined order.
        - **labels** (Optional[str]): JSON array of per-item labels, e.g.
                 ["finance", "people", ""], paired positionally with the data items
                 (one entry per item; an empty entry skips that item). Stored on each
                 item's data record.
        - **external_metadata** (Optional[str]): JSON array of per-item metadata objects,
                 e.g. [{"source": "crm"}, null], paired positionally with the data items
                 (one entry per item; null or {} skips that item). Merged into each
                 item's stored external_metadata.
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
        - **400 Bad Request**: Neither datasetId nor datasetName provided, or neither
          data nor raw_data provided
        - **409 Conflict**: Error during add operation
        - **403 Forbidden**: User doesn't have permission to add to dataset

        ## Notes
        - To add data to datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL is set to True)
        - datasetId value can only be the UUID of an already existing dataset
        """
        # Swagger UI submits an untouched file list as one blank part; treat it
        # as "no uploads" (and reject its "string" placeholder with a clear 400).
        data = drop_blank_uploads(data)

        send_telemetry(
            "Add API Endpoint Invoked",
            user,
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

        # String inputs join the uploads as one item list, uploads first. Drop
        # empty entries — Swagger UI submits untouched array items as "".
        raw_items = [item.strip() for item in (raw_data or []) if item and item.strip()]
        data = [*(data or []), *raw_items]

        # Labels and metadata ride on DataItems, which ingestion unwraps to
        # store them on each item's Data record. Invalid JSON or a count
        # mismatch raises a CogneeApiError (400), returned by the global handler.
        data = pair_labels_with_data(
            data, parse_labels(labels), parse_external_metadata(external_metadata)
        )

        if not data:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="Provide at least one file in 'data' or one entry in 'raw_data'.",
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
                # The failing task's error is carried on ``payload`` (set to
                # ``repr(error)`` by the pipeline runner). Surface it directly so
                # the client gets an actionable message instead of an empty body
                # or the model's repr.
                detail = add_run.payload if isinstance(add_run.payload, str) else None
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content=ErrorResponse(
                        error="Pipeline run errored",
                        detail=detail or str(add_run),
                    ).model_dump(),
                )
            return add_run
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
