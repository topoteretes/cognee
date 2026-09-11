from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, status
from fastapi import UploadFile as UF
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import WithJsonSchema

from cognee import __version__ as cognee_version
from cognee.api.DTO import ErrorResponse
from cognee.api.v1.exceptions import UpdateTargetNotInferredError
from cognee.api.v1.update.result import UpdateBatchResult, UpdateResult
from cognee.exceptions import CogneeApiError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

# NOTE: Needed because of: https://github.com/fastapi/fastapi/discussions/14975
#       Once issue is resolved on Swagger side it can be removed.
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]

logger = get_logger()


def get_update_router() -> APIRouter:
    router = APIRouter()

    @router.patch(
        "",
        response_model=UpdateResult | UpdateBatchResult,
        responses={
            403: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {
                "model": UpdateResult | UpdateBatchResult | ErrorResponse,
                "description": (
                    "The rebuild's cognify run errored (an UpdateResult with status "
                    '"failed", naming the error), every document of a batch failed '
                    '(an UpdateBatchResult with status "failed"), or an unexpected '
                    "error occurred."
                ),
            },
        },
    )
    async def update(
        dataset_id: UUID = Query(
            ...,
            description="UUID of the dataset containing the document to update.",
            examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
        ),
        data: list[UploadFile] = File(
            ...,
            description=(
                "New version of the document that replaces the existing one; several "
                "files update one document each and answer with a batch result. With "
                "chunk_level_diff enabled (default) only the chunks affected by the "
                "edit are replaced; otherwise the document is rebuilt."
            ),
        ),
        data_id: UUID | None = Query(
            default=None,
            description=(
                "UUID of the existing document to update "
                "(returned by GET /api/v1/datasets/{dataset_id}/data). Optional: without "
                "it the document is matched by the upload's filename. Only with one file."
            ),
            examples=["9c4e4a4b-2b1a-4f6e-9d3a-1c2b3d4e5f6a"],
        ),
        node_set: list[str] | None = Form(
            default=[""],
            examples=[["user_memories"]],
            description="Node identifiers for graph organization and access control.",
        ),
        chunk_level_diff: bool = Query(
            default=True,
            description=(
                "Diff the new content against the stored text and re-ingest only the "
                "affected chunks. Falls back to the full rebuild (memory dropped, row refreshed) "
                "when chunk-level preconditions are not met."
            ),
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Update data in a dataset.

        This endpoint updates existing documents in a specified dataset: the new version
        of a document is uploaded as `data`, and the document it replaces is named by
        `data_id` or, without one, matched by the upload's filename among the dataset's
        documents. The document is updated, analyzed, and the changes are integrated
        into the knowledge graph. Several files update one document each.

        ## Request Parameters
        - **dataset_id** (UUID, required, query): UUID of the dataset containing the document to update
        - **data** (List[UploadFile]): New version of the document that replaces the existing one;
                 several files are a batch, each matched by its filename.
        - **data_id** (UUID, optional, query): UUID of the existing document to update (returned by
                 GET /api/v1/datasets/{dataset_id}/data). Required when the filename does not
                 match a stored document (raw text, a renamed file); only with a single file.
        - **node_set** (Optional[List[str]]): List of node identifiers for graph organization and access control.
                 Used for grouping related data points in the knowledge graph.
        - **chunk_level_diff** (bool, query, default true): Replace only the chunks affected
                 by the edit instead of re-ingesting the whole document.

        ## Response
        One body on every path (`UpdateResult`), a superset of the chunk-level summary
        returned before:
        - **status**: `"incremental"` (chunks replaced), `"unchanged"` (no content change),
          `"full_rebuild"` (memory dropped and rebuilt from the new content) or `"failed"` (the rebuild's
          cognify run errored; `error` says why, and the call can be retried).
        - **regions**, **deleted_chunks**, **added_chunks**, **reused_chunks**,
          **kept_chunks**, **reindexed_chunks**, **total_chunks**: the chunk-level
          counters; `null` on a rebuild, which has no diff.
        - **data_id**, **dataset_id**: the document, the handle to retry with.
        - **duration_seconds**: wall-clock time of the update.
        - **pipeline_run_id**: the run to inspect; `null` for a no-op.
        - **fallback**: set on every rebuild; its `reason` names why the chunk-level path
          did not run (`disabled`, `unsupported_metadata`, `custom_extraction_config`,
          `per_call_db_config`, `unsupported_backend`, `unsupported_chunker`,
          `no_baseline`, `chunks_not_tiling`, `unreadable_text`) and `detail` says it in
          a sentence.
        - **error**: `error_class` and `message` when `status` is `"failed"`.

        Several files answer with an `UpdateBatchResult`: `status` (`"completed"`,
        `"partial"` or `"failed"` when every document failed), the counts `total`,
        `updated`, `unchanged`, `failed`, `dataset_id`, `duration_seconds`, and
        `results`, one `UpdateResult` per file in upload order — a file whose update
        raised is recorded there as `"failed"` with the error and the batch carries on.

        ## Error Codes
        - **422 Unprocessable Entity**: dataset_id missing or not a valid UUID; data_id given
          with several files; or no data_id and a file that cannot be matched to a stored
          document (the message says how to list the dataset's documents and pass data_id)
        - **403 Forbidden**: User lacks write permission on the dataset
        - **404 Not Found**: data_id resolves to no document in the dataset
        - **500 Internal Server Error**: the rebuild's cognify run errored (body is the
          `UpdateResult` with status `"failed"`) or an unexpected error occurred

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
            result = await cognee_update(
                # One upload is one document; several are a batch.
                data=data[0] if len(data) == 1 else data,
                dataset_id=dataset_id,
                data_id=data_id,
                user=user,
                node_set=node_set if node_set and node_set != [""] else None,
                chunk_level_diff=chunk_level_diff,
            )

            if result["status"] == "failed":
                # Same body as a success, so the client can read the error and
                # retry this document (or every document of a failed batch);
                # the status code still says it failed.
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content=jsonable_encoder(result),
                )
            return result

        except UpdateTargetNotInferredError as error:
            # The SDK message names SDK calls; HTTP callers get the endpoints.
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content=ErrorResponse(
                    error="Document to update could not be determined; pass data_id",
                    detail=error.api_message,
                ).model_dump(),
            )
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
