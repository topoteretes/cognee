from fastapi.responses import JSONResponse
from fastapi import File, UploadFile as UF, Depends, Form, Query, status
from typing import Optional, Annotated, Dict, Literal, Union
from fastapi import APIRouter
from typing import List
from uuid import UUID
from pydantic import BaseModel, WithJsonSchema
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunErrored,
    PipelineRunInfo,
)
from cognee.api.DTO import ErrorResponse
from cognee.exceptions import CogneeApiError

# NOTE: Needed because of: https://github.com/fastapi/fastapi/discussions/14975
#       Once issue is resolved on Swagger side it can be removed.
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]

logger = get_logger()


class IncrementalUpdateResponse(BaseModel):
    status: Literal["incremental", "unchanged"]
    regions: int
    deleted_chunks: int
    added_chunks: int
    reused_chunks: int
    kept_chunks: int
    reindexed_chunks: int


def get_update_router() -> APIRouter:
    router = APIRouter()

    @router.patch(
        "",
        response_model=Union[IncrementalUpdateResponse, Dict[UUID, PipelineRunInfo]],
        responses={
            403: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def update(
        data_id: UUID = Query(
            ...,
            description=(
                "UUID of the existing document to update "
                "(returned by GET /api/v1/datasets/{dataset_id}/data)."
            ),
            examples=["9c4e4a4b-2b1a-4f6e-9d3a-1c2b3d4e5f6a"],
        ),
        dataset_id: UUID = Query(
            ...,
            description="UUID of the dataset containing the document to update.",
            examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
        ),
        data: List[UploadFile] = File(
            ...,
            description=(
                "New version of the document that replaces the existing one. With "
                "chunk_level_diff enabled (default) only the chunks affected by the "
                "edit are replaced; otherwise the document is deleted and re-ingested."
            ),
        ),
        node_set: Optional[List[str]] = Form(
            default=[""],
            examples=[["user_memories"]],
            description="Node identifiers for graph organization and access control.",
        ),
        chunk_level_diff: bool = Query(
            default=True,
            description=(
                "Diff the new content against the stored text and re-ingest only the "
                "affected chunks. Falls back to the full delete + re-add + cognify flow "
                "when chunk-level preconditions are not met."
            ),
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Update data in a dataset.

        This endpoint updates existing documents in a specified dataset by providing the data_id of the existing document
        to update and the new document with the changes as the data.
        The document is updated, analyzed, and the changes are integrated into the knowledge graph.

        ## Request Parameters
        - **data_id** (UUID, required, query): UUID of the existing document to update (returned by GET /api/v1/datasets/{dataset_id}/data)
        - **dataset_id** (UUID, required, query): UUID of the dataset containing the document to update
        - **data** (List[UploadFile]): New version of the document that replaces the existing one.
        - **node_set** (Optional[List[str]]): List of node identifiers for graph organization and access control.
                 Used for grouping related data points in the knowledge graph.
        - **chunk_level_diff** (bool, query, default true): Replace only the chunks affected
                 by the edit instead of re-ingesting the whole document.

        ## Response
        With chunk_level_diff, a summary of the incremental operation (same keys for
        either status):
        `{"status": "incremental" | "unchanged", "regions": n, "deleted_chunks": n,
        "added_chunks": n, "reused_chunks": n, "kept_chunks": n, "reindexed_chunks": n}`.
        When the full flow runs (chunk_level_diff disabled, or its preconditions not met),
        pipeline run information for the delete + re-add + cognify operation.

        ## Error Codes
        - **422 Unprocessable Entity**: data_id or dataset_id missing or not a valid UUID
        - **403 Forbidden**: User lacks write permission on the dataset
        - **500 Internal Server Error**: Pipeline run errored or an unexpected error occurred during the update

        ## Notes
        - Chunk-level updates keep unaffected chunks, their entities, and their summaries
          untouched; only the edited region is re-extracted.
        """
        send_telemetry(
            "Update API Endpoint Invoked",
            user,
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
                node_set=node_set if node_set and node_set != [""] else None,
                chunk_level_diff=chunk_level_diff,
            )

            # Chunk-level path returns its own summary dict, no pipeline runs.
            if isinstance(update_run, dict) and "status" in update_run:
                return update_run

            # If any cognify run errored return JSONResponse with proper error status code
            if any(isinstance(v, PipelineRunErrored) for v in update_run.values()):
                first_err = next(
                    (v for v in update_run.values() if isinstance(v, PipelineRunErrored)), None
                )
                detail = getattr(first_err, "error", None) if first_err else None
                if not detail:
                    detail = str(first_err) if first_err else "Pipeline run errored"

                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content=ErrorResponse(
                        error="Pipeline run errored",
                        detail=detail,
                    ).model_dump(),
                )
            return update_run

        except CogneeApiError:
            # Typed API errors (e.g. UpdateTargetNotFoundError -> 404) carry
            # their own status codes — let the app-level handler map them
            # instead of flattening everything into a 500.
            raise
        except Exception as error:
            logger.exception("Update failed")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    detail=str(error),
                ).model_dump(),
            )

    return router
