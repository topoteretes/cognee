from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, status
from fastapi import UploadFile as UF
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import WithJsonSchema

from cognee import __version__ as cognee_version
from cognee.api.DTO import ErrorResponse
from cognee.api.v1.update.result import UpdateResult
from cognee.exceptions import CogneeApiError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

# NOTE: Needed because of: https://github.com/fastapi/fastapi/discussions/14975
#       Once issue is resolved on Swagger side it can be removed.
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]

logger = get_logger()

# The form value that clears the node set. A multipart form cannot send an empty
# list, and a single empty field already means "keep" (it is what Swagger UI
# sends for an untouched list field), so clearing gets a literal of its own.
CLEAR_NODE_SET = "[]"


def node_set_from_form(node_set: list[str] | None) -> list[str] | None:
    """The node_set update() receives: None keeps, [] clears, a list replaces."""
    names = [name for name in node_set or [] if name]
    if names == [CLEAR_NODE_SET]:
        return []
    return names or None


def get_update_router() -> APIRouter:
    router = APIRouter()

    @router.patch(
        "",
        response_model=UpdateResult,
        responses={
            403: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {
                "model": UpdateResult | ErrorResponse,
                "description": (
                    "The rebuild's cognify run errored (an UpdateResult with status "
                    '"failed", naming the error) or an unexpected error occurred.'
                ),
            },
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
        data: list[UploadFile] | None = File(
            default=None,
            description=(
                "New version of the document that replaces the existing one. With "
                "chunk_level_diff enabled (default) only the chunks affected by the "
                "edit are replaced; otherwise the document is deleted and re-ingested. "
                "Omit it and send node_set to change only the document's node sets."
            ),
        ),
        node_set: list[str] | None = Form(
            default=[""],
            examples=[["user_memories"]],
            description=(
                "Node identifiers used to organize and filter data points in the graph. "
                "Omit (or send empty) to keep the document's stored node set; passing one "
                "replaces it. The single value `[]` clears it. With data the replacement "
                "runs the full rebuild; without data it is applied in place, without "
                "reprocessing."
            ),
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

        This endpoint updates existing documents in a specified dataset by providing the data_id of the existing document
        to update and the new document with the changes as the data.
        The document is updated, analyzed, and the changes are integrated into the knowledge graph.

        ## Request Parameters
        - **data_id** (UUID, required, query): UUID of the existing document to update (returned by GET /api/v1/datasets/{dataset_id}/data)
        - **dataset_id** (UUID, required, query): UUID of the dataset containing the document to update
        - **data** (Optional[List[UploadFile]]): New version of the document that replaces the
          existing one. Omit it to change only the document's node sets.
        - **node_set** (Optional[List[str]]): List of node identifiers used to organize and filter
          data points in the knowledge graph. Not an access-control mechanism. Omitted (or a
          single empty field) keeps the stored node set, label and metadata of the document
          (the update is a PATCH for document metadata); passing it replaces the node set,
          and the single value `[]` clears it.
        - **chunk_level_diff** (bool, query, default true): Replace only the chunks affected
                 by the edit instead of re-ingesting the whole document.

        ## Response
        One body on every path (`UpdateResult`), a superset of the chunk-level summary
        returned before:
        - **status**: `"incremental"` (chunks replaced), `"unchanged"` (no content change, or
          node sets already as requested), `"metadata_updated"` (node sets replaced in place),
          `"full_rebuild"` (memory dropped and rebuilt from the new content) or `"failed"` (the rebuild's
          cognify run errored; `error` says why, and the call can be retried).
        - **regions**, **deleted_chunks**, **added_chunks**, **reused_chunks**,
          **redated_chunks**, **kept_chunks**, **reindexed_chunks**, **total_chunks**:
          the chunk-level counters; `null` on a rebuild, which has no diff.
        - **data_id**, **dataset_id**: the document, the handle to retry with.
        - **duration_seconds**: wall-clock time of the update.
        - **pipeline_run_id**: the run to inspect; `null` for a no-op.
        - **fallback**: set on every rebuild; its `reason` names why the chunk-level path
          did not run (`disabled`, `unsupported_metadata`, `custom_extraction_config`,
          `per_call_db_config`, `unsupported_backend`, `unsupported_chunker`,
          `no_baseline`, `chunks_not_tiling`, `unreadable_text`) and `detail` says it in
          a sentence.
        - **error**: `error_class` and `message` when `status` is `"failed"`.

        ## Error Codes
        - **400 Bad Request**: neither data nor node_set was sent
        - **422 Unprocessable Entity**: data_id or dataset_id missing or not a valid UUID
        - **403 Forbidden**: User lacks write permission on the dataset
        - **404 Not Found**: data_id resolves to no document in the dataset
        - **500 Internal Server Error**: the rebuild's cognify run errored (body is the
          `UpdateResult` with status `"failed"`) or an unexpected error occurred
        - **501 Not Implemented**: a node_set-only update on storage that cannot change
          memberships in place (checked before anything is written)

        ## Notes
        - Chunk-level updates keep unaffected chunks, their entities, and their summaries
          untouched; only the edited region is re-extracted.
        - A node_set-only update (no data) moves the document's existing memory to the new
          node sets: no LLM call, no re-chunking, no re-embedding.
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
                data_id=data_id,
                data=data,
                dataset_id=dataset_id,
                user=user,
                node_set=node_set_from_form(node_set),
                chunk_level_diff=chunk_level_diff,
            )

            if result["status"] == "failed":
                # Same body as a success, so the client can read the error and
                # retry this document; the status code still says it failed.
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content=jsonable_encoder(result),
                )
            return result

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
