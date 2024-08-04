from duckdb import CatalogException
from cognee.modules.ingestion import discover_directory_datasets
from cognee.modules.tasks import get_task_status
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
    def list_data(dataset_name: str):
        db = get_relational_engine()
        try:
            return db.get_files_metadata(dataset_name)
        except CatalogException:
            return None

    @staticmethod
    def get_status(dataset_ids: list[str]) -> dict:
        try:
            return get_task_status(dataset_ids)
        except CatalogException:
            return {}

    @staticmethod
    def delete_dataset(dataset_id: str):
        db = get_relational_engine()
        try:
            return db.delete_table(dataset_id)
        except CatalogException:
            return {}
