from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.modules.users.methods import get_default_user
from cognee.modules.ingestion import discover_directory_datasets
from cognee.modules.users.models import User
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status


class datasets:
    @staticmethod
    async def list_datasets():
        from cognee.modules.data.methods import get_datasets

        user = await get_default_user()
        return await get_datasets(user.id)

    @staticmethod
    async def get_unique_dataset_id(dataset_name: str, user: User) -> UUID:
        return uuid5(NAMESPACE_OID, f"{dataset_name}{str(user.id)}")

    @staticmethod
    def discover_datasets(directory_path: str):
        return list(discover_directory_datasets(directory_path).keys())

    @staticmethod
    async def list_data(dataset_id: str):
        from cognee.modules.data.methods import get_dataset, get_dataset_data

        user = await get_default_user()

        dataset = await get_dataset(user.id, dataset_id)

        return await get_dataset_data(dataset.id)

    @staticmethod
    async def get_status(dataset_ids: list[UUID]) -> dict:
        return await get_pipeline_status(dataset_ids)

    @staticmethod
    async def delete_dataset(dataset_id: str):
        from cognee.modules.data.methods import get_dataset, delete_dataset

        user = await get_default_user()
        dataset = await get_dataset(user.id, dataset_id)

        return await delete_dataset(dataset)
