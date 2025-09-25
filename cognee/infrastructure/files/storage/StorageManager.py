import inspect
from typing import BinaryIO
from contextlib import asynccontextmanager

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

    storage: Storage = None

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
        if inspect.iscoroutinefunction(self.storage.file_exists):
            return await self.storage.file_exists(file_path)
        else:
            return self.storage.file_exists(file_path)

    async def is_file(self, file_path: str):
        if inspect.iscoroutinefunction(self.storage.is_file):
            return await self.storage.is_file(file_path)
        else:
            return self.storage.is_file(file_path)

    async def get_size(self, file_path: str) -> int:
        if inspect.iscoroutinefunction(self.storage.get_size):
            return await self.storage.get_size(file_path)
        else:
            return self.storage.get_size(file_path)

    async def store(self, file_path: str, data: BinaryIO, overwrite: bool = False) -> str:
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
        if inspect.iscoroutinefunction(self.storage.store):
            return await self.storage.store(file_path, data, overwrite)
        else:
            return self.storage.store(file_path, data, overwrite)

    @asynccontextmanager
    async def open(self, file_path: str, encoding: str = None, *args, **kwargs):
        """
        Retrieve data from the specified file path.

        Parameters:
        -----------

            - file_path (str): The path from which to retrieve the data.

        Returns:
        --------

            Returns the retrieved data, as defined by the storage implementation.
        """
        # Check the actual storage type by class name to determine if open() is async or sync

        if self.storage.__class__.__name__ == "S3FileStorage":
            # S3FileStorage.open() is async
            async with self.storage.open(file_path, *args, **kwargs) as file:
                yield file
        else:
            # LocalFileStorage.open() is sync
            with self.storage.open(file_path, *args, **kwargs) as file:
                yield file

    async def ensure_directory_exists(self, directory_path: str = ""):
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """
        if inspect.iscoroutinefunction(self.storage.ensure_directory_exists):
            return await self.storage.ensure_directory_exists(directory_path)
        else:
            return self.storage.ensure_directory_exists(directory_path)

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
        if inspect.iscoroutinefunction(self.storage.remove):
            return await self.storage.remove(file_path)
        else:
            return self.storage.remove(file_path)

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
        if inspect.iscoroutinefunction(self.storage.list_files):
            return await self.storage.list_files(directory_path, recursive)
        else:
            return self.storage.list_files(directory_path, recursive)

    async def remove_all(self, tree_path: str = None):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        if inspect.iscoroutinefunction(self.storage.remove_all):
            return await self.storage.remove_all(tree_path)
        else:
            return self.storage.remove_all(tree_path)
