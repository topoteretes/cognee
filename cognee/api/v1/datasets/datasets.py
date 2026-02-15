import asyncio
from uuid import UUID
from typing import Optional

from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.data.methods import get_dataset_data, has_dataset_data
from cognee.modules.data.methods import get_authorized_dataset, get_authorized_existing_datasets
from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.graph.methods import (
    delete_data_nodes_and_edges,
    delete_dataset_nodes_and_edges,
    has_data_related_nodes,
    legacy_delete,
)
from cognee.modules.ingestion import discover_directory_datasets
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.shared.logging_utils import get_logger

logger = get_logger()


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
    async def has_data(dataset_id: str, user: Optional[User] = None) -> bool:
        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user.id, dataset_id)

        return await has_dataset_data(dataset.id)

    @staticmethod
    async def get_status(dataset_ids: list[UUID]) -> dict:
        return await get_pipeline_status(dataset_ids, pipeline_name="cognify_pipeline")

    @staticmethod
    async def empty_dataset(dataset_id: UUID, user: Optional[User] = None):
        from cognee.modules.data.methods import delete_data, delete_dataset

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id, "delete")

        if not dataset:
            raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

        await set_database_global_context_variables(dataset.id, dataset.owner_id)

        await delete_dataset_nodes_and_edges(dataset_id, user.id)

        dataset_data = await get_dataset_data(dataset.id)

        # Delete dataset record first while DatasetData junction rows still exist,
        # so pipeline_status cleanup can find related Data records.
        result = await delete_dataset(dataset)

        # Delete individual data records; use return_exceptions so all are attempted
        # even if some fail.
        if dataset_data:
            results = await asyncio.gather(
                *[delete_data(data) for data in dataset_data],
                return_exceptions=True,
            )
            deletion_errors = [r for r in results if isinstance(r, Exception)]
            if deletion_errors:
                logger.error(
                    "Failed to delete %d/%d data items from dataset %s: %s",
                    len(deletion_errors),
                    len(dataset_data),
                    dataset_id,
                    deletion_errors,
                )

        return result

    @staticmethod
    async def delete_data(
        dataset_id: UUID,
        data_id: UUID,
        user: Optional[User] = None,
        mode: str = "soft",  # mode is there for backwards compatibility. Don't use "hard", it is dangerous.
    ):
        from cognee.modules.data.methods import delete_data, get_data

        if not user:
            user = await get_default_user()

        try:
            dataset = await get_authorized_dataset(user, dataset_id, "delete")
        except PermissionDeniedError:
            raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

        if not dataset:
            raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

        dataset_data = [data for data in await get_dataset_data(dataset.id) if data.id == data_id]

        data = dataset_data[0] if len(dataset_data) > 0 else None

        if not data:
            # If data is not found in the system, user is using a custom graph model.
            await set_database_global_context_variables(dataset_id, dataset.owner_id)
            await delete_data_nodes_and_edges(dataset_id, data_id, user.id)
            return {"status": "success"}

        if not any(ds.id == dataset_id for ds in data.datasets):
            raise UnauthorizedDataAccessError(f"Data {data_id} not accessible.")

        await set_database_global_context_variables(dataset_id, dataset.owner_id)

        if not await has_data_related_nodes(dataset_id, data_id):
            await legacy_delete(data, "soft")
        else:
            await delete_data_nodes_and_edges(dataset_id, data_id, user.id)

        await delete_data(data)

        return {"status": "success"}

    @staticmethod
    async def delete_all(user: Optional[User] = None):
        if not user:
            user = await get_default_user()

        user_datasets = await get_authorized_existing_datasets([], "delete", user)

        for dataset in user_datasets:
            await set_database_global_context_variables(dataset.id, dataset.owner_id)

            await datasets.empty_dataset(dataset.id, user)
