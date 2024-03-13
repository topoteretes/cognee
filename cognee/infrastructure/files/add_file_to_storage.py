from typing import BinaryIO
from cognee.root_dir import get_absolute_path
from .storage.StorageManager import StorageManager
from .storage.LocalStorage import LocalStorage

async def add_file_to_storage(file_path: str, file: BinaryIO):
    storage_manager = StorageManager(LocalStorage(get_absolute_path("data/files")))

    storage_manager.store(file_path, file)
