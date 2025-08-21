from .utils import get_storage_type
from .StorageManager import StorageManager
from .storage_provider_registry import StorageProviderRegistry


def get_file_storage(storage_path: str) -> StorageManager:
    provider_name = get_storage_type(storage_path)
    provider_class = StorageProviderRegistry.get(provider_name)

    return StorageManager(provider_class(storage_path))
