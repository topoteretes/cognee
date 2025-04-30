from cognee.shared.logging_utils import get_logger
from fastapi import APIRouter
from datetime import datetime
from uuid import UUID
from typing import List, Optional
from typing_extensions import Annotated
from fastapi import HTTPException, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from cognee.api.DTO import OutDTO
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
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


def get_datasets_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_model=list[DatasetDTO])
    async def get_datasets(user: User = Depends(get_authenticated_user)):
        try:
            from cognee.modules.data.methods import get_datasets

            datasets = await get_datasets(user.id)

            return datasets
        except Exception as error:
            logger.error(f"Error retrieving datasets: {str(error)}")
            raise HTTPException(
                status_code=500, detail=f"Error retrieving datasets: {str(error)}"
            ) from error

    @router.delete(
        "/{dataset_id}", response_model=None, responses={404: {"model": ErrorResponseDTO}}
    )
    async def delete_dataset(dataset_id: UUID, user: User = Depends(get_authenticated_user)):
        from cognee.modules.data.methods import get_dataset, delete_dataset

        dataset = await get_dataset(user.id, dataset_id)

        if dataset is None:
            raise EntityNotFoundError(message=f"Dataset ({str(dataset_id)}) not found.")

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

        # TODO: Handle situation differently if user doesn't have permission to access data?
        if dataset is None:
            raise EntityNotFoundError(message=f"Dataset ({str(dataset_id)}) not found.")

        data = await get_data(user.id, data_id)

        if data is None:
            raise EntityNotFoundError(message=f"Data ({str(data_id)}) not found.")

        await delete_data(data)

    @router.get("/{dataset_id}/graph", response_model=str)
    async def get_dataset_graph(dataset_id: UUID, user: User = Depends(get_authenticated_user)):
        from cognee.shared.utils import render_graph
        from cognee.infrastructure.databases.graph import get_graph_engine

        try:
            graph_client = await get_graph_engine()
            graph_url = await render_graph(graph_client.graph)

            return JSONResponse(
                status_code=200,
                content=str(graph_url),
            )
        except Exception as error:
            print(error)
            return JSONResponse(
                status_code=409,
                content="Graphistry credentials are not set. Please set them in your .env file.",
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
            raise EntityNotFoundError(message=f"No data found in dataset ({dataset_id}).")

        matching_data = [data for data in dataset_data if data.id == data_id]

        # Check if matching_data contains an element
        if len(matching_data) == 0:
            raise EntityNotFoundError(
                message=f"Data ({data_id}) not found in dataset ({dataset_id})."
            )

        data = await get_data(user.id, data_id)

        if data is None:
            raise EntityNotFoundError(
                message=f"Data ({data_id}) not found in dataset ({dataset_id})."
            )

        return data.raw_data_location

    return router
