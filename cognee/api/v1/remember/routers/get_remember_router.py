from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import Form, File, UploadFile as UF, Depends
from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field, WithJsonSchema

from cognee.memory import QAEntry, TraceEntry, FeedbackEntry
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version

logger = get_logger()

# NOTE: Needed because of: https://github.com/fastapi/fastapi/discussions/14975
#       Once issue is resolved on Swagger side it can be removed.
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]


def get_remember_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    @log_usage(function_name="POST /v1/remember", log_type="api_endpoint")
    async def remember(
        data: List[UploadFile] = File(default=None),
        datasetName: Optional[str] = Form(default=None),
        datasetId: Union[UUID, Literal[""], None] = Form(default=None, examples=[""]),
        session_id: Optional[str] = Form(default=None, examples=[""]),
        node_set: Optional[List[str]] = Form(default=[""], example=[""]),
        run_in_background: Optional[bool] = Form(default=False),
        custom_prompt: Optional[str] = Form(default=""),
        chunks_per_batch: Optional[int] = Form(default=10),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Ingest data and build the knowledge graph in a single call.

        This endpoint combines the add and cognify steps. Data is ingested
        first, then automatically processed into a structured knowledge graph.

        ## Request Parameters
        - **data** (List[UploadFile]): Files to upload and process.
        - **datasetName** (Optional[str]): Name of the target dataset.
        - **datasetId** (Optional[UUID]): UUID of an existing dataset.
        - **node_set** (Optional[List[str]]): Node identifiers for graph organisation.
        - **run_in_background** (Optional[bool]): Run the cognify step asynchronously (default: False).
        - **custom_prompt** (Optional[str]): Custom prompt for entity extraction.
        - **chunks_per_batch** (Optional[int]): Chunks per cognify batch.

        Either datasetName or datasetId must be provided.

        ## Error Codes
        - **400 Bad Request**: Neither datasetId nor datasetName provided
        - **409 Conflict**: Error during processing
        """
        send_telemetry(
            "Remember API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/remember",
                "node_set": node_set,
                "cognee_version": cognee_version,
            },
        )

        if not datasetId and not datasetName:
            raise HTTPException(
                status_code=400,
                detail="Either datasetId or datasetName must be provided.",
            )

        from cognee.api.v1.remember import remember as cognee_remember

        try:
            result = await cognee_remember(
                data,
                dataset_name=datasetName,
                session_id=session_id,
                user=user,
                dataset_id=datasetId if datasetId else None,
                node_set=node_set if node_set != [""] else None,
                run_in_background=run_in_background or False,
                custom_prompt=custom_prompt or None,
                chunks_per_batch=chunks_per_batch,
            )

            return jsonable_encoder(result.to_dict())
        except Exception as error:
            logger.error("Remember endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during remember."},
            )

    class RememberEntryRequest(BaseModel):
        """JSON body for the typed-entry remember endpoint.

        ``entry`` is a discriminated union — set ``type`` to ``qa``,
        ``trace``, or ``feedback`` and include the corresponding fields.
        """

        entry: Annotated[
            Union[QAEntry, TraceEntry, FeedbackEntry],
            Field(discriminator="type"),
        ]
        dataset_name: str = "main_dataset"
        session_id: str

    @router.post("/entry", response_model=dict)
    @log_usage(function_name="POST /v1/remember/entry", log_type="api_endpoint")
    async def remember_entry(
        payload: RememberEntryRequest,
        user: User = Depends(get_authenticated_user),
    ):
        """Store a typed memory entry in the session cache.

        Accepts a discriminated union of ``QAEntry``, ``TraceEntry``, or
        ``FeedbackEntry`` and dispatches to the matching SessionManager
        method. Always requires ``session_id``.

        ## Response
        The returned ``RememberResult`` includes ``entry_type`` and
        ``entry_id`` — the ``qa_id``/``trace_id`` returned by the cache
        (or the ``qa_id`` a feedback was attached to). Use this to chain
        feedback to a freshly stored QA.
        """
        send_telemetry(
            "Remember Entry API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/remember/entry",
                "entry_type": payload.entry.type,
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.remember import remember as cognee_remember

        try:
            result = await cognee_remember(
                payload.entry,
                dataset_name=payload.dataset_name,
                session_id=payload.session_id,
                user=user,
            )
            return jsonable_encoder(result.to_dict())
        except ValueError as error:
            # Known validation errors: missing session_id, user not found, etc.
            return JSONResponse(status_code=400, content={"error": str(error)})
        except RuntimeError as error:
            # Session cache unavailable
            return JSONResponse(status_code=503, content={"error": str(error)})
        except Exception as error:
            logger.error("Remember entry endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during remember."},
            )

    return router
