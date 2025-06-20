from typing import BinaryIO
from contextlib import contextmanager

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

    def file_exists(self, file_path: str):
        """
        Check if a specified file exists in the storage.

        Parameters:
        -----------

            - file_path (str): The path of the file to check for existence.

        Returns:
        --------

            - bool: True if the file exists, otherwise False.
        """
        return self.storage.file_exists(file_path)

    def store(self, file_path: str, data: BinaryIO):
        """
        Store data at the specified file path.

        Parameters:
        -----------

            - file_path (str): The path where the data should be stored.
            - data (BinaryIO): The data in a binary format that needs to be stored.

        Returns:
        --------

            Returns the outcome of the store operation, as defined by the storage
            implementation.
        """
        return self.storage.store(file_path, data)

    @contextmanager
    def open(self, file_path: str, *args, **kwargs):
        """
        Retrieve data from the specified file path.

        Parameters:
        -----------

            - file_path (str): The path from which to retrieve the data.

        Returns:
        --------

            Returns the retrieved data, as defined by the storage implementation.
        """
        with self.storage.open(file_path, *args, **kwargs) as file:
            yield file

    def ensure_directory_exists(self, directory_path: str = None):
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """
        return self.storage.ensure_directory_exists(directory_path)

    def remove(self, file_path: str):
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
        return self.storage.remove(file_path)

    def remove_all(self, tree_path: str = None):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        return self.storage.remove_all(tree_path)
