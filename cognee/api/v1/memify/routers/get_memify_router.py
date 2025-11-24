from uuid import UUID

from fastapi import APIRouter
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
from cognee import __version__ as cognee_version

logger = get_logger()


class MemifyPayloadDTO(InDTO):
    extraction_tasks: Optional[List[str]] = Field(
        default=None,
        examples=[[]],
    )
    enrichment_tasks: Optional[List[str]] = Field(default=None, examples=[[]])
    data: Optional[str] = Field(default="")
    dataset_name: Optional[str] = Field(default=None)
    # Note: Literal is needed for Swagger use
    dataset_id: Union[UUID, Literal[""], None] = Field(default=None, examples=[""])
    node_name: Optional[List[str]] = Field(default=None, examples=[[]])
    run_in_background: Optional[bool] = Field(default=False)


def get_memify_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    async def memify(payload: MemifyPayloadDTO, user: User = Depends(get_authenticated_user)):
        """
        Enrichment pipeline in Cognee, can work with already built graphs. If no data is provided existing knowledge graph will be used as data,
        custom data can also be provided instead which can be processed with provided extraction and enrichment tasks.

        Provided tasks and data will be arranged to run the Cognee pipeline and execute graph enrichment/creation.

        ## Request Parameters
        - **extractionTasks** Optional[List[str]]: List of available Cognee Tasks to execute for graph/data extraction.
        - **enrichmentTasks** Optional[List[str]]: List of available Cognee Tasks to handle enrichment of provided graph/data from extraction tasks.
        - **data** Optional[List[str]]: The data to ingest. Can be any text data when custom extraction and enrichment tasks are used.
              Data provided here will be forwarded to the first extraction task in the pipeline as input.
              If no data is provided the whole graph (or subgraph if node_name/node_type is specified) will be forwarded
        - **dataset_name** (Optional[str]): Name of the datasets to memify
        - **dataset_id** (Optional[UUID]): List of UUIDs of an already existing dataset
        - **node_name** (Optional[List[str]]):  Filter graph to specific named entities (for targeted search). Used when no data is provided.
        - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to False (blocking).

        Either datasetName or datasetId must be provided.

        ## Response
        Returns information about the add operation containing:
        - Status of the operation
        - Details about the processed data
        - Any relevant metadata from the ingestion process

        ## Error Codes
        - **400 Bad Request**: Neither datasetId nor datasetName provided
        - **409 Conflict**: Error during memify operation
        - **403 Forbidden**: User doesn't have permission to use dataset

        ## Notes
        - To memify datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL is set to True)
        - datasetId value can only be the UUID of an already existing dataset
        """

        send_telemetry(
            "Memify API Endpoint Invoked",
            user.id,
            additional_properties={"endpoint": "POST /v1/memify", "cognee_version": cognee_version},
        )

        if not payload.dataset_id and not payload.dataset_name:
            raise ValueError("Either datasetId or datasetName must be provided.")

        try:
            from cognee.modules.memify import memify as cognee_memify

            memify_run = await cognee_memify(
                extraction_tasks=payload.extraction_tasks,
                enrichment_tasks=payload.enrichment_tasks,
                data=payload.data,
                dataset=payload.dataset_id if payload.dataset_id else payload.dataset_name,
                node_name=payload.node_name,
                user=user,
            )

            if isinstance(memify_run, PipelineRunErrored):
                return JSONResponse(status_code=420, content=memify_run)
            return memify_run
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
