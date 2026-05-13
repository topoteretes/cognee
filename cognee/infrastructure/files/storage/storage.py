from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import BinaryIO, Protocol

from cognee.infrastructure.files.storage.FileBufferedReader import FileBufferedReader


class Storage(Protocol):
    storage_path: str

    """
    Abstract interface for storage operations.
    """

    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a specified file exists in the storage.

        Parameters:
        -----------

            - file_path (str): The path of the file to check for existence.

        Returns:
        --------

            - bool: True if the file exists, otherwise False.
        """
        pass

    async def is_file(self, file_path: str) -> bool:
        """
        Check if a specified file is a regular file.

        Parameters:
        -----------

            - file_path (str): The path of the file to check.

        Returns:
        --------

            - bool: True if the file is a regular file, otherwise False.
        """
        pass

    async def get_size(self, file_path: str) -> int:
        """
        Get the size of a specified file in bytes.

        Parameters:
        -----------

            - file_path (str): The path of the file to get the size of.

        Returns:
        --------

            - int: The size of the file in bytes.
        """
        pass

    async def store(self, file_path: str, data: BinaryIO | str, overwrite: bool):
        """
        Store data at the specified file path.

        Parameters:
        -----------

            - file_path (str): The path where the data will be stored.
            - data (bytes): The binary data to be stored.
            - overwrite (bool): If True, overwrite the existing file.
        """
        pass

    @asynccontextmanager
    def open(self, file_path: str, mode: str = "r") -> AsyncGenerator[FileBufferedReader]:
        """
        Retrieve file from the specified file path.

        Parameters:
        -----------

            - file_path (str): The path from where the data will be retrieved.
            - mode (str): The mode to open the file, with "r" as the default for reading text
        """
        pass

    async def copy_file(self, source_file_path: str, destination_file_path: str) -> str:
        """
        Copy a file from a source path to a destination path.

        Parameters:
        -----------

            - source_file_path (str): The path of the file to be copied.
            - destination_file_path (str): The path where the file will be copied to.

        Returns:
        --------

            - str: The path to the copied file.
        """
        pass

    async def ensure_directory_exists(self, directory_path: str = "") -> None:
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """
        pass

    async def remove(self, file_path: str) -> None:
        """
        Remove the storage at the specified file path.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        pass

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
        pass

    async def remove_all(self, root_path: str | None = None) -> None:
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        pass
