import inspect
from contextlib import asynccontextmanager
from typing import BinaryIO

from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

from .storage import Storage


class StorageManager:
    """
    Manages storage operations by delegating tasks to a storage backend.

    Public methods include:
    - store: Store data in the specified path.
    - open: Open a file from the specified path.
    - remove: Remove the file at the specified path.
    - remove_all: Remove all files under the directory tree.
    """

    def __init__(self, storage: Storage):
        self.storage = storage

    async def file_exists(self, file_path: str):
        """
        Check if a specified file exists in the storage.

        Parameters:
        -----------

            - file_path (str): The path of the file to check for existence.

        Returns:
        --------

            - bool: True if the file exists, otherwise False.
        """
        return await self.storage.file_exists(file_path)

    async def is_file(self, file_path: str):
        return await self.storage.is_file(file_path)

    async def get_size(self, file_path: str) -> int:
        return await self.storage.get_size(file_path)

    async def store(self, file_path: str, data: BinaryIO | str, overwrite: bool = False) -> str:
        """
        Store data at the specified file path.

        Parameters:
        -----------

            - file_path (str): The path where the data should be stored.
            - data (BinaryIO): The data in a binary format that needs to be stored.
            - overwrite (bool): If True, overwrite the existing file.

        Returns:
        --------

            Returns the full path to the file.
        """
        return await self.storage.store(file_path, data, overwrite)

    @asynccontextmanager
    async def open(self, file_path: str, mode: str = "r", *args, **kwargs):
        """
        Retrieve data from the specified file path.

        Parameters:
        -----------

            - file_path (str): The path from which to retrieve the data.

        Returns:
        --------

            Returns the retrieved data, as defined by the storage implementation.
        """
        async with self.storage.open(file_path, mode, *args, **kwargs) as file:
            yield file

    async def ensure_directory_exists(self, directory_path: str = ""):
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """
        return await self.storage.ensure_directory_exists(directory_path)

    async def remove(self, file_path: str):
        """
        Remove the file at the specified path.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.

        Returns:
        --------

            Returns the outcome of the remove operation, as defined by the storage
            implementation.
        """
        return await self.storage.remove(file_path)

    async def list_files(self, directory_path: str, recursive: bool = False) -> list[str]:
        """
        List all files in the specified directory.

        Parameters:
        -----------
            - directory_path (str): The directory path to list files from
            - recursive (bool): If True, list files recursively in subdirectories

        Returns:
        --------
            - list[str]: List of file paths relative to the storage root
        """
        return await self.storage.list_files(directory_path, recursive)

    async def remove_all(self, tree_path: str | None = None) -> None:
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """

        await self.storage.remove_all(tree_path)
