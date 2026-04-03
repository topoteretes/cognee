from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import Form, File, UploadFile, Depends
from typing import List, Optional, Union, Literal

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version

logger = get_logger()


def get_remember_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    @log_usage(function_name="POST /v2/remember", log_type="api_endpoint")
    async def remember(
        data: List[UploadFile] = File(default=None),
        datasetName: Optional[str] = Form(default=None),
        datasetId: Union[UUID, Literal[""], None] = Form(default=None, examples=[""]),
        node_set: Optional[List[str]] = Form(default=[""], example=[""]),
        run_in_background: Optional[bool] = Form(default=False),
        custom_prompt: Optional[str] = Form(default=""),
        chunks_per_batch: Optional[int] = Form(default=None),
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
                "endpoint": "POST /v2/remember",
                "node_set": node_set,
                "cognee_version": cognee_version,
            },
        )

        if not datasetId and not datasetName:
            raise ValueError("Either datasetId or datasetName must be provided.")

        from cognee.api.v2.remember import remember as cognee_remember

        try:
            result = await cognee_remember(
                data,
                dataset_name=datasetName or "main_dataset",
                user=user,
                dataset_id=datasetId if datasetId else None,
                node_set=node_set if node_set != [""] else None,
                run_in_background=run_in_background or False,
                custom_prompt=custom_prompt or None,
                chunks_per_batch=chunks_per_batch,
            )

            return result
        except Exception as error:
            logger.error("Remember endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during remember."},
            )

    return router
