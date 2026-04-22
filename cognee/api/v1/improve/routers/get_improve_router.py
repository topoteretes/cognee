from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Depends
from pydantic import Field
from typing import List, Optional, Union, Literal

from cognee.api.DTO import InDTO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.modules.pipelines.models import PipelineRunErrored
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version

logger = get_logger()


class ImprovePayloadDTO(InDTO):
    extraction_tasks: Optional[List[str]] = Field(default=None, examples=[[]])
    enrichment_tasks: Optional[List[str]] = Field(default=None, examples=[[]])
    data: Optional[str] = Field(default="")
    dataset_name: Optional[str] = Field(default=None)
    dataset_id: Union[UUID, Literal[""], None] = Field(default=None, examples=[""])
    node_name: Optional[List[str]] = Field(default=None, examples=[[]])
    run_in_background: Optional[bool] = Field(default=False)
    # Session IDs to bridge into the permanent graph. When set, improve
    # runs the full session pipeline (feedback weights + QA persist +
    # trace-step persist + graph→session sync) in addition to the
    # default memify enrichment.
    session_ids: Optional[List[str]] = Field(default=None, examples=[[]])


def get_improve_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    @log_usage(function_name="POST /v1/improve", log_type="api_endpoint")
    async def improve(payload: ImprovePayloadDTO, user: User = Depends(get_authenticated_user)):
        """
        Enrich and improve the knowledge graph.

        This is a memory-oriented alias for the memify endpoint. It runs
        enrichment tasks on an existing knowledge graph.

        ## Request Parameters
        - **extraction_tasks** (Optional[List[str]]): Tasks for graph/data extraction.
        - **enrichment_tasks** (Optional[List[str]]): Tasks for graph enrichment.
        - **data** (Optional[str]): Custom input data. Uses existing graph when empty.
        - **dataset_name** (Optional[str]): Dataset name.
        - **dataset_id** (Optional[UUID]): Dataset UUID.
        - **node_name** (Optional[List[str]]): Filter to specific named entities.
        - **run_in_background** (Optional[bool]): Run asynchronously (default: False).

        Either dataset_name or dataset_id must be provided.

        ## Error Codes
        - **400 Bad Request**: Neither dataset_id nor dataset_name provided
        - **409 Conflict**: Error during processing
        """
        send_telemetry(
            "Improve API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/improve",
                "cognee_version": cognee_version,
            },
        )

        if not payload.dataset_id and not payload.dataset_name:
            raise HTTPException(
                status_code=400,
                detail="Either datasetId or datasetName must be provided.",
            )

        try:
            from cognee.api.v1.improve import improve as cognee_improve

            improve_run = await cognee_improve(
                extraction_tasks=payload.extraction_tasks,
                enrichment_tasks=payload.enrichment_tasks,
                data=payload.data,
                dataset=payload.dataset_id if payload.dataset_id else payload.dataset_name,
                node_name=payload.node_name,
                session_ids=payload.session_ids,
                user=user,
                run_in_background=payload.run_in_background,
            )

            if isinstance(improve_run, PipelineRunErrored):
                return JSONResponse(status_code=420, content=improve_run)
            return improve_run
        except Exception as error:
            logger.error("Improve endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during graph improvement."},
            )

    return router
