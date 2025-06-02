from typing import BinaryIO
from cognee.root_dir import get_absolute_path
from .storage.StorageManager import StorageManager
from .storage.LocalStorage import LocalStorage


async def add_file_to_storage(file_path: str, file: BinaryIO):
    """
    Store a file in local storage.

    This function initializes a storage manager and uses it to store the provided file at
    the specified file path.

    Parameters:
    -----------

        - file_path (str): The path where the file will be stored.
        - file (BinaryIO): The file object to be stored, which must be a binary file.
    """
    storage_manager = StorageManager(LocalStorage(get_absolute_path("data/files")))

    storage_manager.store(file_path, file)
