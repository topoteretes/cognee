from cognee.root_dir import get_absolute_path
from .storage.StorageManager import StorageManager
from .storage.LocalStorage import LocalStorage


async def remove_file_from_storage(file_path: str):
    """
    Remove a specified file from storage.

    This function initializes a storage manager with a local storage instance and calls the
    remove method of the storage manager to delete the file identified by the given path.
    Ensure that the file exists in the specified storage before calling this function to
    avoid
    potential errors.

    Parameters:
    -----------

        - file_path (str): The path of the file to remove from storage.
    """
    storage_manager = StorageManager(LocalStorage(get_absolute_path("data/files")))

    storage_manager.remove(file_path)
