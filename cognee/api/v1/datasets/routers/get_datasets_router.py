from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from typing_extensions import Annotated
from fastapi import status
from fastapi import APIRouter
from fastapi import HTTPException, Query, Depends
from fastapi.responses import JSONResponse, FileResponse

from cognee.api.DTO import InDTO, OutDTO
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import create_dataset, get_datasets_by_name
from cognee.shared.logging_utils import get_logger
from cognee.api.v1.delete.exceptions import DataNotFoundError, DatasetNotFoundError
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.permissions.methods import (
    get_all_user_permission_datasets,
    give_permission_on_dataset,
)
from cognee.modules.graph.methods import get_formatted_graph_data
from cognee.modules.pipelines.models import PipelineRunStatus

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
        dataset_data: DatasetCreationPayload, user: User = Depends(get_authenticated_user)
    ):
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
        try:
            from cognee.modules.data.methods import get_dataset

            dataset = await get_dataset(user.id, dataset_id)

            formatted_graph_data = await get_formatted_graph_data(dataset.id, user.id)

            return JSONResponse(
                status_code=200,
                content=formatted_graph_data,
            )
        except Exception:
            return JSONResponse(
                status_code=409,
                content="Error retrieving dataset graph data.",
            )

    @router.get(
        "/{dataset_id}/data",
        response_model=list[DataDTO],
        responses={404: {"model": ErrorResponseDTO}},
    )
    async def get_dataset_data(dataset_id: UUID, user: User = Depends(get_authenticated_user)):
        from cognee.modules.data.methods import get_dataset_data, get_dataset

        dataset = await get_dataset(user.id, dataset_id)

        if dataset is None:
            return JSONResponse(
                status_code=404,
                content=ErrorResponseDTO(f"Dataset ({str(dataset_id)}) not found."),
            )

        dataset_data = await get_dataset_data(dataset_id=dataset.id)

        if dataset_data is None:
            return []

        return dataset_data

    @router.get("/status", response_model=dict[str, PipelineRunStatus])
    async def get_dataset_status(
        datasets: Annotated[List[UUID], Query(alias="dataset")] = None,
        user: User = Depends(get_authenticated_user),
    ):
        from cognee.api.v1.datasets.datasets import datasets as cognee_datasets

        try:
            datasets_statuses = await cognee_datasets.get_status(datasets)

            return datasets_statuses
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.get("/{dataset_id}/data/{data_id}/raw", response_class=FileResponse)
    async def get_raw_data(
        dataset_id: UUID, data_id: UUID, user: User = Depends(get_authenticated_user)
    ):
        from cognee.modules.data.methods import get_data
        from cognee.modules.data.methods import get_dataset, get_dataset_data

        dataset = await get_dataset(user.id, dataset_id)

        if dataset is None:
            return JSONResponse(
                status_code=404, content={"detail": f"Dataset ({dataset_id}) not found."}
            )

        dataset_data = await get_dataset_data(dataset.id)

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
