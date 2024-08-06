from duckdb import CatalogException
from cognee.modules.ingestion import discover_directory_datasets
from cognee.modules.data.operations.get_dataset_data import get_dataset_data
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.infrastructure.databases.relational import get_relational_engine

class datasets():
    @staticmethod
    async def list_datasets():
        db = get_relational_engine()
        return await db.get_datasets()

    @staticmethod
    def discover_datasets(directory_path: str):
        return list(discover_directory_datasets(directory_path).keys())

    @staticmethod
    async def list_data(dataset_id: str, dataset_name: str = None):
        try:
            return await get_dataset_data(dataset_id = dataset_id, dataset_name = dataset_name)
        except CatalogException:
            return None

    @staticmethod
    async def get_status(dataset_ids: list[str]) -> dict:
        try:
            return await get_pipeline_status(dataset_ids)
        except CatalogException:
            return {}

    @staticmethod
    async def delete_dataset(dataset_id: str):
        db = get_relational_engine()
        try:
            return await db.delete_table(dataset_id)
        except CatalogException:
            return {}
