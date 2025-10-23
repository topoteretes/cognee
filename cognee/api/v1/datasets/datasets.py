from uuid import UUID
from cognee.modules.data.methods import has_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.modules.ingestion import discover_directory_datasets
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status


class datasets:
    @staticmethod
    async def list_datasets():
        from cognee.modules.data.methods import get_datasets

        user = await get_default_user()
        return await get_datasets(user.id)

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
    async def has_data(dataset_id: str) -> bool:
        from cognee.modules.data.methods import get_dataset

        user = await get_default_user()

        dataset = await get_dataset(user.id, dataset_id)

        return await has_dataset_data(dataset.id)

    @staticmethod
    async def get_status(dataset_ids: list[UUID]) -> dict:
        return await get_pipeline_status(dataset_ids, pipeline_name="cognify_pipeline")

    @staticmethod
    async def delete_dataset(dataset_id: str):
        from cognee.modules.data.methods import get_dataset, delete_dataset

        user = await get_default_user()
        dataset = await get_dataset(user.id, dataset_id)

        return await delete_dataset(dataset)
