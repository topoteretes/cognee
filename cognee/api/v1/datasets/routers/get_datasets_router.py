from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Annotated
from fastapi import status
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException, Query, Depends
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from urllib.parse import urlparse
from pathlib import Path

from cognee import datasets
from cognee.api.DTO import InDTO, OutDTO
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.data.methods import get_datasets_graph_counts
from cognee.modules.data.methods.create_authorized_dataset import create_authorized_dataset
from cognee.shared.logging_utils import get_logger
from cognee.api.v1.exceptions import DataNotFoundError
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets
from cognee.modules.graph.methods import get_formatted_graph_data
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

logger = get_logger()


class ErrorResponseDTO(BaseModel):
    message: str


# Shared by GET /status and GET /status/progress — both accept the same
# dataset/pipeline selection, so the query param definitions (alias,
# description, examples) live here once instead of twice.
StatusDatasetIdsQuery = Annotated[
    List[UUID],
    Query(
        alias="dataset",
        description=(
            "Dataset UUIDs to check (from GET /api/v1/datasets)."
            " Omit to get status for all datasets you can read."
        ),
        examples=[["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"]],
    ),
]

StatusPipelineNamesQuery = Annotated[
    List[str],
    Query(
        alias="pipeline",
        description=(
            "Pipeline names to check: 'add_pipeline', 'cognify_pipeline', or"
            " 'code_graph_pipeline' (code ingestion via remember"
            " content_type='code'). Omit to default to cognify_pipeline."
        ),
        examples=[["cognify_pipeline"]],
    ),
]


class PipelineRunStatusWithProgress(BaseModel):
    status: PipelineRunStatus
    # Present only once a run has emitted at least one progress tick (see
    # log_pipeline_run_progress); None before that or for terminal runs that
    # predate this field.
    progress: Optional[Dict[str, Any]] = Field(
        default=None,
        examples=[{"completed_items": 3, "total_items": 10, "current_stage": "extract_graph"}],
    )


class DatasetDTO(OutDTO):
    id: UUID
    name: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    owner_id: UUID


class DataDTO(OutDTO):
    id: UUID
    name: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    extension: str
    mime_type: str
    raw_data_location: str
    dataset_id: UUID
    label: Optional[str] = None
    external_metadata: Optional[dict] = None


class DatasetGraphSummaryDTO(OutDTO):
    dataset_id: UUID
    pipeline_run_id: Optional[UUID] = None
    num_nodes: int
    num_edges: int
    # None while pipeline_run_id is set means the last count attempt degraded
    # (graph store unavailable) and wasn't cached — retried on the next poll.
    computed_at: Optional[datetime] = None


class GraphNodeDTO(OutDTO):
    id: UUID
    label: str
    type: str
    properties: dict


class GraphEdgeDTO(OutDTO):
    source: UUID
    target: UUID
    label: str


class GraphDTO(OutDTO):
    nodes: List[GraphNodeDTO]
    edges: List[GraphEdgeDTO]


class DatasetCreationPayload(InDTO):
    name: str = Field(
        examples=["main_dataset"],
        description=(
            "Name of the dataset to create. If a dataset with this name already exists"
            " for the user, the existing dataset is returned instead of creating a duplicate."
        ),
    )


class DatasetSchemaPayloadDTO(InDTO):
    graph_schema: Optional[Dict[str, Any]] = None
    custom_prompt: Optional[str] = None


def get_datasets_router() -> APIRouter:
    router = APIRouter()

    @router.get("", response_model=list[DatasetDTO])
    async def get_datasets(user: User = Depends(get_authenticated_user)):
        """
        Get all datasets accessible to the authenticated user.

        This endpoint retrieves all datasets that the authenticated user has
        read permissions for. The datasets are returned with their metadata
        including ID, name, creation time, and owner information.

        ## Response
        Returns a list of dataset objects containing:
        - **id**: Unique dataset identifier
        - **name**: Dataset name
        - **created_at**: When the dataset was created
        - **updated_at**: When the dataset was last updated
        - **owner_id**: ID of the dataset owner

        ## Error Codes
        - **500 Internal Server Error**: Error retrieving datasets
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/datasets",
                "cognee_version": cognee_version,
            },
        )

        try:
            datasets = await get_all_user_permission_datasets(user, "read")

            return datasets
        except Exception as error:
            logger.error(f"Error retrieving datasets: {str(error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error retrieving datasets: {str(error)}",
            ) from error

    @router.post("", response_model=DatasetDTO)
    async def create_new_dataset(
        dataset_data: DatasetCreationPayload,
        user: User = Depends(get_authenticated_user),
    ):
        """
        Create a new dataset or return existing dataset with the same name.

        This endpoint creates a new dataset with the specified name. If a dataset
        with the same name already exists for the user, it returns the existing
        dataset instead of creating a duplicate. The user is automatically granted
        all permissions (read, write, share, delete) on the created dataset.

        ## Request Parameters
        - **dataset_data** (DatasetCreationPayload): Dataset creation parameters containing:
          - **name**: The name for the new dataset

        ## Response
        Returns the created or existing dataset object containing:
        - **id**: Unique dataset identifier
        - **name**: Dataset name
        - **created_at**: When the dataset was created
        - **updated_at**: When the dataset was last updated
        - **owner_id**: ID of the dataset owner

        ## Error Codes
        - **500 Internal Server Error**: Error creating dataset
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "POST /v1/datasets",
                "cognee_version": cognee_version,
            },
        )

        try:
            datasets = await get_datasets_by_name([dataset_data.name], user.id)

            if datasets:
                return datasets[0]

            dataset = await create_authorized_dataset(dataset_data.name, user)

            return dataset
        except Exception as error:
            logger.error(f"Error creating dataset: {str(error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating dataset: {str(error)}",
            ) from error

    @router.delete("")
    async def delete_all(user: User = Depends(get_authenticated_user)):
        """
        Delete all user's data.

        This endpoint permanently deletes all datasets that user created and all its associated data.
        The user must have delete permissions on the dataset to perform this operation.

        ## Response
        No content returned on successful deletion.
        If no datasets exist for the users, nothing happens.
        """
        await datasets.delete_all(user)

    @router.delete(
        "/{dataset_id}", response_model=None, responses={404: {"model": ErrorResponseDTO}}
    )
    async def delete_dataset(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Delete a dataset by its ID.

        This endpoint permanently deletes a dataset and all its associated data.
        The user must have delete permissions on the dataset to perform this operation.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset to delete

        ## Response
        No content returned on successful deletion.

        ## Error Codes
        - **401/403 Unauthorized/Forbidden**: Dataset doesn't exist or user lacks delete permission
        - **500 Internal Server Error**: Error during deletion
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"DELETE /v1/datasets/{str(dataset_id)}",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        await datasets.empty_dataset(dataset_id, user)

    @router.delete(
        "/{dataset_id}/data/{data_id}",
        response_model=None,
        responses={404: {"model": ErrorResponseDTO}},
    )
    async def delete_data(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        data_id: UUID = PathParam(
            description="Data item UUID, from GET /api/v1/datasets/{dataset_id}/data",
            examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Delete a specific data item from a dataset.

        This endpoint removes a specific data item from a dataset while keeping
        the dataset itself intact. The user must have delete permissions on the
        dataset to perform this operation.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset containing the data
        - **data_id** (UUID): The unique identifier of the data item to delete

        ## Response
        No content returned on successful deletion.

        ## Error Codes
        - **401 Unauthorized**: Dataset doesn't exist or user lacks delete permission
        - **500 Internal Server Error**: Error during deletion

        ## Notes
        Deleting a data_id not tracked in the dataset is treated as a custom-graph-model
        deletion and returns success.
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"DELETE /v1/datasets/{str(dataset_id)}/data/{str(data_id)}",
                "dataset_id": str(dataset_id),
                "data_id": str(data_id),
                "cognee_version": cognee_version,
            },
        )

        await datasets.delete_data(dataset_id, data_id, user)

    @router.get("/{dataset_id}/graph", response_model=GraphDTO)
    async def get_dataset_graph(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get the knowledge graph visualization for a dataset.

        This endpoint retrieves the knowledge graph data for a specific dataset,
        including nodes and edges that represent the relationships between entities
        in the dataset. The graph data is formatted for visualization purposes.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset

        ## Response
        Returns the graph data containing:
        - **nodes**: List of graph nodes with id, label, type, and properties
        - **edges**: List of graph edges with source, target, and label

        ## Error Codes
        - **404 Not Found**: Dataset doesn't exist or user doesn't have access
        - **500 Internal Server Error**: Error retrieving graph data
        """

        graph_data = await get_formatted_graph_data(dataset_id, user)

        return graph_data

    @router.get(
        "/{dataset_id}/data",
        response_model=list[DataDTO],
        responses={404: {"model": ErrorResponseDTO}},
    )
    async def get_dataset_data(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get all data items in a dataset.

        This endpoint retrieves all data items (documents, files, etc.) that belong
        to a specific dataset. Each data item includes metadata such as name, type,
        creation time, and storage location.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset

        ## Response
        Returns a list of data objects containing:
        - **id**: Unique data item identifier
        - **name**: Data item name
        - **created_at**: When the data was added
        - **updated_at**: When the data was last updated
        - **extension**: File extension
        - **mime_type**: MIME type of the data
        - **raw_data_location**: Storage location of the raw data
        - **dataset_id**: ID of the containing dataset
        - **label**: Label attached to the data item at upload, if any
        - **external_metadata**: Stored metadata dict (upload-provided keys merged over
          loader-derived ones), if any

        ## Error Codes
        - **404 Not Found**: Dataset doesn't exist or user doesn't have access
        - **500 Internal Server Error**: Error retrieving data
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"GET /v1/datasets/{str(dataset_id)}/data",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.data.methods import get_dataset_data

        # Verify user has permission to read dataset
        dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

        if not dataset:
            return JSONResponse(
                status_code=404,
                content=ErrorResponseDTO(
                    message=f"Dataset ({str(dataset_id)}) not found."
                ).model_dump(),
            )

        dataset_id = dataset[0].id

        dataset_data = await get_dataset_data(dataset_id=dataset_id)

        if dataset_data is None:
            return []

        # Dict literal, not dict(**data, dataset_id=...): Data now carries its
        # own dataset_id column, and the kwarg form raises TypeError on the
        # duplicate key. The requested dataset id still wins — the column is
        # nullable, so the row's value cannot be relied on here.
        return [
            {
                **jsonable_encoder(data),
                "dataset_id": dataset_id,
            }
            for data in dataset_data
        ]

    @router.get(
        "/status",
        response_model=Union[dict[str, PipelineRunStatus], dict[str, dict[str, PipelineRunStatus]]],
    )
    async def get_dataset_status(
        datasets: StatusDatasetIdsQuery = [],
        pipelines: StatusPipelineNamesQuery = [],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get the processing status of datasets.

        This endpoint retrieves the current processing status of one or more datasets,
        indicating whether they are being processed, have completed processing, or
        encountered errors during pipeline execution.

        ## Query Parameters
        - **dataset** (List[UUID]): List of dataset UUIDs to check status for.
          If omitted, returns status for all datasets the user has read permission on
        - **pipeline** (List[str], optional): One or more pipeline names to check.
          - If omitted, defaults to **cognify_pipeline** (backward-compatible behavior)
          - If one pipeline is provided, response is a flat map
          - If multiple pipelines are provided, response is nested per dataset and pipeline
          - **Available options: add_pipeline, cognify_pipeline, code_graph_pipeline**
          - Note: a background code ingest creates its pipeline run only once the
            repository is cloned — a dataset missing from the response means the run
            has not started yet, not that it failed

        ## Response
        Returns status information in one of two shapes:
        - Single pipeline (default): {dataset_id: status}
        - Multiple pipelines: {dataset_id: {pipeline_name: status}}

        Status values:
        - **pending**: Dataset is queued for processing
        - **running**: Dataset is currently being processed
        - **completed**: Dataset processing completed successfully
        - **failed**: Dataset processing encountered an error

        For in-flight progress (files completed / total, current stage), see
        **GET /v1/datasets/status/progress** — a separate endpoint with its own
        fixed response shape, rather than a flag here that would change what
        this endpoint returns depending on how it's called.

        ## Error Codes
        - **409 Conflict**: Error retrieving status (e.g. requesting a dataset you don't have
          read permission for)
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/datasets/status",
                "datasets": [str(dataset_id) for dataset_id in datasets],
                "pipelines": pipelines,
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.datasets.datasets import datasets as cognee_datasets

        try:
            # Verify user has permission to read dataset
            authorized_datasets = await get_authorized_existing_datasets(datasets, "read", user)

            datasets_statuses = await cognee_datasets.get_status(
                [dataset.id for dataset in authorized_datasets],
                pipeline_names=pipelines or None,
            )

            return datasets_statuses
        except Exception as error:
            logger.error("Error retrieving dataset statuses: %s", error)
            return JSONResponse(
                status_code=409,
                content={"error": "Unable to retrieve dataset statuses."},
            )

    @router.get(
        "/status/progress",
        response_model=Union[
            dict[str, PipelineRunStatusWithProgress],
            dict[str, dict[str, PipelineRunStatusWithProgress]],
        ],
    )
    async def get_dataset_progress(
        datasets: StatusDatasetIdsQuery = [],
        pipelines: StatusPipelineNamesQuery = [],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get the processing status of datasets, together with in-flight progress.

        Same dataset/pipeline selection as **GET /v1/datasets/status**, but each
        status value is always an object {status, progress} instead of a bare
        status — a dedicated endpoint rather than a flag on /status, so neither
        endpoint's response shape ever depends on how it was called.

        ## Query Parameters
        - **dataset** (List[UUID]): Dataset UUIDs to check (from GET /api/v1/datasets). Omit to get
          status for all datasets you can read.
        - **pipeline** (List[str]): Pipeline names to check: 'add_pipeline', 'cognify_pipeline', or
          'code_graph_pipeline' (code ingestion via remember content_type='code'). Omit to default
          to cognify_pipeline.

        ## Response
        - Single pipeline (default): {dataset_id: {status, progress}}
        - Multiple pipelines: {dataset_id: {pipeline_name: {status, progress}}}

        **progress** is `null` until the first in-flight progress tick, then an
        object with `completed_items`, `total_items`, and `current_stage` —
        present only while the pipeline is running; terminal runs (completed/
        errored) do not carry a progress snapshot.

        ## Error Codes
        - **409 Conflict**: Error retrieving status (e.g. requesting a dataset you don't have
          read permission for)
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/datasets/status/progress",
                "datasets": [str(dataset_id) for dataset_id in datasets],
                "pipelines": pipelines,
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.datasets.datasets import datasets as cognee_datasets

        try:
            # Verify user has permission to read dataset
            authorized_datasets = await get_authorized_existing_datasets(datasets, "read", user)

            datasets_progress = await cognee_datasets.get_progress(
                [dataset.id for dataset in authorized_datasets],
                pipeline_names=pipelines or None,
            )

            return datasets_progress
        except Exception as error:
            logger.error("Error retrieving dataset progress: %s", error)
            return JSONResponse(
                status_code=409,
                content={"error": "Unable to retrieve dataset progress."},
            )

    @router.get("/graph-summary", response_model=List[DatasetGraphSummaryDTO])
    async def get_datasets_graph_summary(
        dataset_ids: Annotated[
            List[UUID],
            Query(
                alias="dataset_ids",
                description=(
                    "Dataset UUIDs to summarize (from GET /api/v1/datasets)."
                    " Omit to summarize every dataset you can read."
                ),
                examples=[["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"]],
            ),
        ] = [],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get node/edge counts per dataset, cached per cognify run.

        Counts are computed once per dataset's latest cognify run and cached in
        GraphMetrics, keyed by pipeline_run_id — orders of magnitude cheaper on
        repeat polls than GET /{dataset_id}/graph, which does a full traversal.

        ## Query Parameters
        - **dataset_ids** (List[UUID], optional): Dataset UUIDs to summarize.
          If omitted, summarizes every dataset the user has read permission on.

        ## Response
        Returns a list of summaries containing:
        - **datasetId**: The dataset's UUID
        - **pipelineRunId**: The dataset's latest cognify run, or null if it
          has never been cognified
        - **numNodes** / **numEdges**: Graph size for that run
        - **computedAt**: When the count was cached, or null when it wasn't —
          either the last attempt degraded (graph store unavailable, counts
          are 0 and retried on the next poll) or a concurrent caller cached
          the same run first (counts are exact)

        ## Error Codes
        - **409 Conflict**: The summary could not be built (generic message;
          the detail is server-logged rather than returned). A single
          unreadable graph store does not cause this — that dataset comes back
          with zero counts — so this means the relational read itself failed.
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/datasets/graph-summary",
                "dataset_ids": [str(dataset_id) for dataset_id in dataset_ids],
                "cognee_version": cognee_version,
            },
        )

        try:
            authorized_datasets = await get_authorized_existing_datasets(dataset_ids, "read", user)

            if not authorized_datasets:
                return []

            counts = await get_datasets_graph_counts(authorized_datasets)
        except Exception as error:
            # Same posture as GET /statuses above and the sibling
            # GET /visualize/brains-summary: a poll that fails transiently is a
            # 409 with a generic message, not an unhandled 500 carrying
            # internals to the client. Scoped to the two relational reads
            # this route makes; the DTO construction below is pure Python
            # over an already-validated shape, so a bug there still surfaces
            # as a real 500 instead of being misreported as this endpoint's
            # documented transient-failure case.
            logger.error("Error retrieving dataset graph summary: %s", error)
            return JSONResponse(
                status_code=409,
                content={"error": "Unable to retrieve dataset graph summary."},
            )

        return [
            DatasetGraphSummaryDTO(
                dataset_id=dataset.id,
                pipeline_run_id=counts[dataset.id].pipeline_run_id,
                num_nodes=counts[dataset.id].num_nodes,
                num_edges=counts[dataset.id].num_edges,
                computed_at=counts[dataset.id].computed_at,
            )
            for dataset in authorized_datasets
        ]

    @router.get("/{dataset_id}/data/{data_id}/raw", response_class=FileResponse)
    async def get_raw_data(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        data_id: UUID = PathParam(
            description="Data item UUID, from GET /api/v1/datasets/{dataset_id}/data",
            examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
        ),
        user: User = Depends(get_authenticated_user),
    ) -> Response:
        """
        Download the raw data file for a specific data item.

        This endpoint allows users to download the original, unprocessed data file
        for a specific data item within a dataset. The file is returned as a direct
        download with appropriate headers.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset containing the data
        - **data_id** (UUID): The unique identifier of the data item to download

        ## Response
        Returns the raw data file as a downloadable response.

        ## Error Codes
        - **404 Not Found**: Data item doesn't exist in the dataset, or its raw file is missing
        - **500 Internal Server Error**: Error accessing the raw data file
        - **501 Not Implemented**: Raw data is stored on an unsupported storage scheme
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"GET /v1/datasets/{str(dataset_id)}/data/{str(data_id)}/raw",
                "dataset_id": str(dataset_id),
                "data_id": str(data_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.data.methods import get_dataset_data, resolve_data_id

        # Verify user has permission to read dataset
        dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

        if not dataset:
            return JSONResponse(
                status_code=404, content={"message": f"Dataset ({dataset_id}) not found."}
            )

        # Dataset-scoped lookup: resolves the exact id or, for a row whose
        # identity forked in the dataset-scoping upgrade, its recorded
        # pre-fork legacy_id — every id ever issued keeps resolving.
        resolved_id = await resolve_data_id(dataset[0].id, data_id)

        if resolved_id is None:
            raise DataNotFoundError(
                message=f"Data ({data_id}) not found in dataset ({dataset_id})."
            )

        matching_data = [
            data for data in await get_dataset_data(dataset[0].id) if data.id == resolved_id
        ]

        if len(matching_data) == 0:
            raise DataNotFoundError(
                message=f"Data ({data_id}) not found in dataset ({dataset_id})."
            )

        # Use the data object already verified to belong to the authorized dataset,
        # rather than calling get_data() which checks owner_id and would reject
        # ACL-granted readers who are not the data owner.
        data = matching_data[0]

        raw_location = data.raw_data_location
        parsed_uri = urlparse(raw_location)

        if parsed_uri.scheme == "s3":
            from cognee.infrastructure.files.utils.open_data_file import open_data_file
            from cognee.infrastructure.utils.run_async import run_async

            download_name = Path(parsed_uri.path).name or data.name
            media_type = data.mime_type or "application/octet-stream"

            async def file_iterator(chunk_size: int = 1024 * 1024):
                async with open_data_file(raw_location, mode="rb") as file:
                    while True:
                        chunk = await run_async(file.read, chunk_size)
                        if not chunk:
                            break
                        yield chunk

            return StreamingResponse(
                file_iterator(),
                media_type=media_type,
                headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
            )

        if parsed_uri.scheme in ("file", "") or (
            len(parsed_uri.scheme) == 1 and parsed_uri.scheme.isalpha()
        ):
            from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path

            file_path = get_data_file_path(raw_location)
            path = Path(file_path)

            if not path.is_file():
                raise DataNotFoundError(message=f"Raw file not found on disk for data ({data_id}).")

            return FileResponse(path=path)

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Storage scheme '{parsed_uri.scheme}' not supported for direct download.",
        )

    @router.get("/{dataset_id}/schema", response_model=dict)
    async def get_dataset_schema(dataset_id: UUID, user: User = Depends(get_authenticated_user)):
        """Return the stored graph schema and custom prompt for a dataset.

        ## Path Parameters
        - **dataset_id** (UUID): UUID of the dataset (from GET /api/v1/datasets).
        """
        from cognee.modules.data.models import DatasetConfiguration
        from sqlalchemy import select

        dataset = await get_authorized_existing_datasets([dataset_id], "read", user)
        if not dataset:
            return JSONResponse(status_code=404, content={"error": "Dataset not found"})

        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            config = await session.scalar(
                select(DatasetConfiguration).where(DatasetConfiguration.dataset_id == dataset_id)
            )
        if not config:
            return {"graph_schema": None, "custom_prompt": None}
        return {
            "graph_schema": config.graph_schema,
            "custom_prompt": config.custom_prompt,
        }

    @router.put("/{dataset_id}/schema", response_model=dict)
    async def update_dataset_schema(
        dataset_id: UUID,
        payload: DatasetSchemaPayloadDTO,
        user: User = Depends(get_authenticated_user),
    ):
        """Store or update the graph schema and custom prompt for a dataset.

        ## Path Parameters
        - **dataset_id** (UUID): UUID of the dataset (from GET /api/v1/datasets).

        ## Request Parameters
        - **customPrompt** (Optional[str]): Custom extraction prompt to store for the
          dataset; omitting it leaves any existing prompt unchanged.
        - **graphSchema** (Optional[Dict[str, Any]]): JSON graph schema to store for the
          dataset; omitting it leaves any existing schema unchanged.
        """
        from cognee.modules.data.models import DatasetConfiguration
        from sqlalchemy import select

        dataset = await get_authorized_existing_datasets([dataset_id], "write", user)
        if not dataset:
            return JSONResponse(status_code=404, content={"error": "Dataset not found"})

        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            config = await session.scalar(
                select(DatasetConfiguration).where(DatasetConfiguration.dataset_id == dataset_id)
            )
            if config:
                if payload.graph_schema is not None:
                    config.graph_schema = payload.graph_schema
                if payload.custom_prompt is not None:
                    config.custom_prompt = payload.custom_prompt
            else:
                config = DatasetConfiguration(
                    dataset_id=dataset_id,
                    graph_schema=payload.graph_schema,
                    custom_prompt=payload.custom_prompt,
                )
                session.add(config)
            await session.commit()
        return {"status": "ok"}

    return router
