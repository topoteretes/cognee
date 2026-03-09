from datetime import datetime
from typing import List, Literal, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cognee.api.DTO import InDTO
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.pipelines.methods import get_pipeline_run, get_pipeline_run_by_dataset
from cognee.modules.pipelines.models import PipelineRun
from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.modules.pipelines.models import PipelineRunErrored
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version

logger = get_logger()
MEMIFY_PIPELINE_NAME = "memify_pipeline"


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


class MemifyStatusDTO(BaseModel):
    pipeline_run_id: UUID
    dataset_id: UUID
    pipeline_name: str
    status: str
    created_at: datetime
    run_info: Optional[dict] = None
    dataset_name: Optional[str] = None


def _sanitize_run_info(run_info: Optional[dict]) -> Optional[dict]:
    if not isinstance(run_info, dict):
        return None

    sanitized_run_info = {key: value for key, value in run_info.items() if key != "data"}
    return sanitized_run_info or None


def _serialize_pipeline_run(
    pipeline_run: PipelineRun, dataset_name: Optional[str] = None
) -> MemifyStatusDTO:
    status = pipeline_run.status
    if isinstance(status, PipelineRunStatus):
        status = status.value

    run_info = _sanitize_run_info(pipeline_run.run_info)

    return MemifyStatusDTO(
        pipeline_run_id=pipeline_run.pipeline_run_id,
        dataset_id=pipeline_run.dataset_id,
        pipeline_name=pipeline_run.pipeline_name,
        status=status,
        created_at=pipeline_run.created_at,
        run_info=run_info,
        dataset_name=dataset_name,
    )


def get_memify_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    @log_usage(function_name="POST /v1/memify", log_type="api_endpoint")
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
                run_in_background=payload.run_in_background,
            )

            if isinstance(memify_run, PipelineRunErrored):
                return JSONResponse(status_code=420, content=memify_run)
            return memify_run
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.get("/status", response_model=MemifyStatusDTO)
    @log_usage(function_name="GET /v1/memify/status", log_type="api_endpoint")
    async def get_latest_memify_status(
        dataset_name: Optional[str] = None,
        dataset_id: Optional[UUID] = None,
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get the latest memify pipeline run for a dataset.

        Use this endpoint when you know the dataset and want the most recent memify run for it.
        For exact status tracking of a previously started background run, prefer the pipeline_run_id endpoint.

        ## Query Parameters
        - **dataset_name** (Optional[str]): Dataset name to resolve for the authenticated user.
        - **dataset_id** (Optional[UUID]): Existing dataset UUID to inspect.

        Exactly one of dataset_name or dataset_id must be provided.

        ## Response
        Returns the latest memify pipeline run record for the dataset, including:
        - **pipeline_run_id**: Stable identifier for one specific memify execution.
        - **status**: Latest persisted pipeline status for that run.
        - **run_info**: Additional run metadata captured when the pipeline was logged.

        ## Notes
        A dataset can have multiple memify runs over time, so dataset-based status only returns the latest one.
        Use pipeline_run_id when a client must continue polling the same execution it started earlier.
        """
        send_telemetry(
            "Memify API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/memify/status",
                "dataset_id": str(dataset_id) if dataset_id else None,
                "dataset_name": dataset_name,
                "cognee_version": cognee_version,
            },
        )

        if bool(dataset_name) == bool(dataset_id):
            return JSONResponse(
                status_code=400,
                content={"error": "Provide exactly one of dataset_name or dataset_id."},
            )

        dataset_selector = dataset_id if dataset_id else dataset_name

        try:
            authorized_datasets = await get_authorized_existing_datasets(
                [dataset_selector], "read", user
            )
            if not authorized_datasets:
                return JSONResponse(status_code=404, content={"error": "Dataset not found."})
            if dataset_name and len(authorized_datasets) > 1:
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "Multiple readable datasets match dataset_name. Use dataset_id instead."
                    },
                )

            pipeline_run = await get_pipeline_run_by_dataset(
                authorized_datasets[0].id, MEMIFY_PIPELINE_NAME
            )
            if pipeline_run is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": "No memify pipeline run found for dataset."},
                )

            return _serialize_pipeline_run(
                pipeline_run, dataset_name=authorized_datasets[0].name
            )
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.get("/status/{pipeline_run_id}", response_model=MemifyStatusDTO)
    @log_usage(function_name="GET /v1/memify/status/{pipeline_run_id}", log_type="api_endpoint")
    async def get_memify_status_by_run_id(
        pipeline_run_id: UUID, user: User = Depends(get_authenticated_user)
    ):
        """
        Get the status of one specific memify pipeline run.

        This endpoint is the precise polling target for asynchronous memify requests because
        pipeline_run_id identifies one concrete execution even when a dataset has been memified multiple times.

        ## Path Parameters
        - **pipeline_run_id** (UUID): Identifier returned when memify starts in background mode.

        ## Response
        Returns the persisted memify pipeline run record with status and run_info metadata.

        ## Error Codes
        - **404 Not Found**: Pipeline run does not exist, is not a memify run, or the user cannot read its dataset.
        - **409 Conflict**: Error while loading the pipeline run.
        """
        send_telemetry(
            "Memify API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": f"GET /v1/memify/status/{str(pipeline_run_id)}",
                "pipeline_run_id": str(pipeline_run_id),
                "cognee_version": cognee_version,
            },
        )

        try:
            pipeline_run = await get_pipeline_run(pipeline_run_id)
            if pipeline_run is None or pipeline_run.pipeline_name != MEMIFY_PIPELINE_NAME:
                return JSONResponse(status_code=404, content={"error": "Memify pipeline run not found."})

            authorized_datasets = await get_authorized_existing_datasets(
                [pipeline_run.dataset_id], "read", user
            )
            if not authorized_datasets:
                return JSONResponse(status_code=404, content={"error": "Memify pipeline run not found."})

            return _serialize_pipeline_run(
                pipeline_run, dataset_name=authorized_datasets[0].name
            )
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
