from cognee.modules.discovery import discover_directory_datasets
from cognee.infrastructure import infrastructure_config

class datasets():
    @staticmethod
    def list_datasets():
        db = infrastructure_config.get_config("database_engine")
        return db.get_datasets()

    @staticmethod
    def discover_datasets(directory_path: str):
        return list(discover_directory_datasets(directory_path).keys())

    @staticmethod
    def query_data(dataset_name: str):
        db = infrastructure_config.get_config("database_engine")
        return db.get_files_metadata(dataset_name)
