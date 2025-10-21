from typing import Optional
from uuid import UUID
from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.data.methods import get_authorized_dataset, get_authorized_existing_datasets
from cognee.modules.graph.methods import delete_data_nodes_and_edges, delete_dataset_nodes_and_edges
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_default_user
from cognee.modules.ingestion import discover_directory_datasets
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.users.models import User


class datasets:
    @staticmethod
    async def list_datasets(user: Optional[User] = None):
        if user is None:
            user = await get_default_user()

        return await get_authorized_existing_datasets([], "read", user)

    @staticmethod
    def discover_datasets(directory_path: str):
        return list(discover_directory_datasets(directory_path).keys())

    @staticmethod
    async def list_data(dataset_id: UUID, user: Optional[User] = None):
        from cognee.modules.data.methods import get_dataset_data

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id)

        return await get_dataset_data(dataset.id)

    @staticmethod
    async def get_status(dataset_ids: list[UUID]) -> dict:
        return await get_pipeline_status(dataset_ids, pipeline_name="cognify_pipeline")

    @staticmethod
    async def delete_dataset(dataset_id: UUID, user: Optional[User] = None):
        from cognee.modules.data.methods import delete_dataset

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id, "delete")

        if not dataset:
            raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

        await delete_dataset_nodes_and_edges(dataset_id)

        return await delete_dataset(dataset)

    @staticmethod
    async def delete_data(dataset_id: UUID, data_id: UUID, user: Optional[User] = None):
        from cognee.modules.data.methods import delete_data, get_data

        if not user:
            user = await get_default_user()

        try:
            await get_authorized_dataset(user, dataset_id, "delete")
        except PermissionDeniedError:
            raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

        data = await get_data(user.id, data_id)

        if not data:
            # If data is not found in the system, user is using a custom graph model.
            await delete_data_nodes_and_edges(dataset_id, data_id)
            return

        data_datasets = data.datasets

        if not data or not any([dataset.id == dataset_id for dataset in data_datasets]):
            raise UnauthorizedDataAccessError(f"Data {data_id} not accessible.")

        await delete_data_nodes_and_edges(dataset_id, data.id)

        await delete_data(data)

    @staticmethod
    async def delete_all(user: Optional[User] = None):
        if not user:
            user = await get_default_user()

        user_datasets = await get_authorized_existing_datasets([], "read", user)

        for dataset in user_datasets:
            await datasets.delete_dataset(dataset.id, user)
