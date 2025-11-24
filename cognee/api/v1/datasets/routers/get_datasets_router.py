from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from typing_extensions import Annotated
from fastapi import status
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException, Query, Depends
from fastapi.responses import JSONResponse, FileResponse

from cognee.api.DTO import InDTO, OutDTO
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods import create_dataset, get_datasets_by_name
from cognee.shared.logging_utils import get_logger
from cognee.api.v1.exceptions import DataNotFoundError, DatasetNotFoundError
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.permissions.methods import (
    get_all_user_permission_datasets,
    give_permission_on_dataset,
)
from cognee.modules.graph.methods import get_formatted_graph_data
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

logger = get_logger()


class ErrorResponseDTO(BaseModel):
    message: str


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


class GraphNodeDTO(OutDTO):
    id: UUID
    label: str
    properties: dict


class GraphEdgeDTO(OutDTO):
    source: UUID
    target: UUID
    label: str


class GraphDTO(OutDTO):
    nodes: List[GraphNodeDTO]
    edges: List[GraphEdgeDTO]


class DatasetCreationPayload(InDTO):
    name: str


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
        - **418 I'm a teapot**: Error retrieving datasets
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user.id,
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
                status_code=status.HTTP_418_IM_A_TEAPOT,
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
        - **418 I'm a teapot**: Error creating dataset
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/datasets",
                "cognee_version": cognee_version,
            },
        )

        try:
            datasets = await get_datasets_by_name([dataset_data.name], user.id)

            if datasets:
                return datasets[0]

            db_engine = get_relational_engine()
            async with db_engine.get_async_session() as session:
                dataset = await create_dataset(
                    dataset_name=dataset_data.name, user=user, session=session
                )

                await give_permission_on_dataset(user, dataset.id, "read")
                await give_permission_on_dataset(user, dataset.id, "write")
                await give_permission_on_dataset(user, dataset.id, "share")
                await give_permission_on_dataset(user, dataset.id, "delete")

                return dataset
        except Exception as error:
            logger.error(f"Error creating dataset: {str(error)}")
            raise HTTPException(
                status_code=status.HTTP_418_IM_A_TEAPOT,
                detail=f"Error creating dataset: {str(error)}",
            ) from error

    @router.delete(
        "/{dataset_id}", response_model=None, responses={404: {"model": ErrorResponseDTO}}
    )
    async def delete_dataset(dataset_id: UUID, user: User = Depends(get_authenticated_user)):
        """
        Delete a dataset by its ID.

        This endpoint permanently deletes a dataset and all its associated data.
        The user must have delete permissions on the dataset to perform this operation.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset to delete

        ## Response
        No content returned on successful deletion.

        ## Error Codes
        - **404 Not Found**: Dataset doesn't exist or user doesn't have access
        - **500 Internal Server Error**: Error during deletion
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": f"DELETE /v1/datasets/{str(dataset_id)}",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.data.methods import get_dataset, delete_dataset

        dataset = await get_dataset(user.id, dataset_id)

        if dataset is None:
            raise DatasetNotFoundError(message=f"Dataset ({str(dataset_id)}) not found.")

        await delete_dataset(dataset)

    @router.delete(
        "/{dataset_id}/data/{data_id}",
        response_model=None,
        responses={404: {"model": ErrorResponseDTO}},
    )
    async def delete_data(
        dataset_id: UUID, data_id: UUID, user: User = Depends(get_authenticated_user)
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
        - **404 Not Found**: Dataset or data item doesn't exist, or user doesn't have access
        - **500 Internal Server Error**: Error during deletion
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": f"DELETE /v1/datasets/{str(dataset_id)}/data/{str(data_id)}",
                "dataset_id": str(dataset_id),
                "data_id": str(data_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.data.methods import get_data, delete_data
        from cognee.modules.data.methods import get_dataset

        # Check if user has permission to access dataset and data by trying to get the dataset
        dataset = await get_dataset(user.id, dataset_id)

        if dataset is None:
            raise DatasetNotFoundError(message=f"Dataset ({str(dataset_id)}) not found.")

        data = await get_data(user.id, data_id)

        if data is None:
            raise DataNotFoundError(message=f"Data ({str(data_id)}) not found.")

        await delete_data(data)

    @router.get("/{dataset_id}/graph", response_model=GraphDTO)
    async def get_dataset_graph(dataset_id: UUID, user: User = Depends(get_authenticated_user)):
        """
        Get the knowledge graph visualization for a dataset.

        This endpoint retrieves the knowledge graph data for a specific dataset,
        including nodes and edges that represent the relationships between entities
        in the dataset. The graph data is formatted for visualization purposes.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset

        ## Response
        Returns the graph data containing:
        - **nodes**: List of graph nodes with id, label, and properties
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
    async def get_dataset_data(dataset_id: UUID, user: User = Depends(get_authenticated_user)):
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

        ## Error Codes
        - **404 Not Found**: Dataset doesn't exist or user doesn't have access
        - **500 Internal Server Error**: Error retrieving data
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": f"GET /v1/datasets/{str(dataset_id)}/data",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.data.methods import get_dataset_data

        # Verify user has permission to read dataset
        dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

        if dataset is None:
            return JSONResponse(
                status_code=404,
                content=ErrorResponseDTO(f"Dataset ({str(dataset_id)}) not found."),
            )

        dataset_id = dataset[0].id

        dataset_data = await get_dataset_data(dataset_id=dataset_id)

        if dataset_data is None:
            return []

        return [
            dict(
                **jsonable_encoder(data),
                dataset_id=dataset_id,
            )
            for data in dataset_data
        ]

    @router.get("/status", response_model=dict[str, PipelineRunStatus])
    async def get_dataset_status(
        datasets: Annotated[List[UUID], Query(alias="dataset")] = [],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get the processing status of datasets.

        This endpoint retrieves the current processing status of one or more datasets,
        indicating whether they are being processed, have completed processing, or
        encountered errors during pipeline execution.

        ## Query Parameters
        - **dataset** (List[UUID]): List of dataset UUIDs to check status for

        ## Response
        Returns a dictionary mapping dataset IDs to their processing status:
        - **pending**: Dataset is queued for processing
        - **running**: Dataset is currently being processed
        - **completed**: Dataset processing completed successfully
        - **failed**: Dataset processing encountered an error

        ## Error Codes
        - **500 Internal Server Error**: Error retrieving status information
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/datasets/status",
                "datasets": [str(dataset_id) for dataset_id in datasets],
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.datasets.datasets import datasets as cognee_datasets

        try:
            # Verify user has permission to read dataset
            authorized_datasets = await get_authorized_existing_datasets(datasets, "read", user)

            datasets_statuses = await cognee_datasets.get_status(
                [dataset.id for dataset in authorized_datasets]
            )

            return datasets_statuses
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.get("/{dataset_id}/data/{data_id}/raw", response_class=FileResponse)
    async def get_raw_data(
        dataset_id: UUID, data_id: UUID, user: User = Depends(get_authenticated_user)
    ):
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
        - **404 Not Found**: Dataset or data item doesn't exist, or user doesn't have access
        - **500 Internal Server Error**: Error accessing the raw data file
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": f"GET /v1/datasets/{str(dataset_id)}/data/{str(data_id)}/raw",
                "dataset_id": str(dataset_id),
                "data_id": str(data_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.data.methods import get_data
        from cognee.modules.data.methods import get_dataset_data

        # Verify user has permission to read dataset
        dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

        if dataset is None:
            return JSONResponse(
                status_code=404, content={"detail": f"Dataset ({dataset_id}) not found."}
            )

        dataset_data = await get_dataset_data(dataset[0].id)

        if dataset_data is None:
            raise DataNotFoundError(message=f"No data found in dataset ({dataset_id}).")

        matching_data = [data for data in dataset_data if data.id == data_id]

        # Check if matching_data contains an element
        if len(matching_data) == 0:
            raise DataNotFoundError(
                message=f"Data ({data_id}) not found in dataset ({dataset_id})."
            )

        data = await get_data(user.id, data_id)

        if data is None:
            raise DataNotFoundError(
                message=f"Data ({data_id}) not found in dataset ({dataset_id})."
            )

        return data.raw_data_location

    return router
