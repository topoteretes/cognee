from cognee.root_dir import get_absolute_path
from .storage.StorageManager import StorageManager
from .storage.LocalStorage import LocalStorage


async def remove_file_from_storage(file_path: str):
    storage_manager = StorageManager(LocalStorage(get_absolute_path("data/files")))

    storage_manager.remove(file_path)
