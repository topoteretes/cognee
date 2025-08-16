from .StorageManager import StorageManager
from .registry import StorageProviderRegistry
from .storage_config import get_cloud_storage_config


def get_file_storage(storage_path: str) -> StorageManager:
    config = get_cloud_storage_config()
    storage_backend = config.get("storage_backend")

    if storage_backend == "local":
        from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage

        return StorageManager(LocalFileStorage(storage_path))

    else:
        provider_name = config.get(storage_backend)
        provider_class = StorageProviderRegistry.get(provider_name)
        return StorageManager(provider_class(storage_path))
