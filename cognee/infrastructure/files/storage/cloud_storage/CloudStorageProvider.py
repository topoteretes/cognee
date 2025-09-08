import os
from abc import abstractmethod, ABC
from typing import BinaryIO, Union, Optional
from contextlib import asynccontextmanager
from fsspec.asyn import AsyncFileSystem

from cognee.infrastructure.utils.run_async import run_async
from cognee.infrastructure.files.storage.FileBufferedReader import FileBufferedReader
from ..storage import Storage


class CloudStorageProvider(Storage, ABC):
    """
    Abstract interface for cloud storage operations.
    """

    storage_path: str
    fs: AsyncFileSystem

    def __init__(self, storage_path: str):
        """Initializes the cloud storage provider and the underlying filesystem."""
        self.storage_path = storage_path
        self.fs = self._initialize_filesystem()

    @abstractmethod
    def _initialize_filesystem(self) -> AsyncFileSystem:
        """
        Initializes and returns the specific filesystem object for the cloud provider (e.g., S3FileSystem).
        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def scheme(self) -> str:
        """
        Returns the URL scheme for the cloud provider (e.g., "s3://", "gs://").
        This property must be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_full_path(self, relative_path: str) -> str:
        """Constructs the full, scheme-less path for the filesystem."""
        return os.path.join(self.storage_path.replace(self.scheme, ""), relative_path)

    async def store(
        self, file_path: str, data: Union[BinaryIO, str], overwrite: bool = False
    ) -> str:
        """
        Store data into a specified file path. The data can be either a string or a binary
        stream.

        This method ensures that the storage directory exists before attempting to write the
        data. If the provided data is a stream, it reads from the stream and writes to the file;
        otherwise, it directly writes the provided data.

        Parameters:
        -----------

            - file_path (str): The relative path of the file where the data will be stored.
            - data (Union[BinaryIO, str]): The data to be stored, which can be a string or a
              binary stream.
            - overwrite (bool): If True, overwrite the existing file.
        """
        full_file_path = self._get_full_path(file_path)

        file_dir_path = os.path.dirname(full_file_path)

        await self.ensure_directory_exists(file_dir_path)

        if overwrite or not await self.file_exists(file_path):

            def save_data_to_file():
                with self.fs.open(
                    full_file_path,
                    mode="w" if isinstance(data, str) else "wb",
                    encoding="utf-8" if isinstance(data, str) else None,
                ) as file:
                    if hasattr(data, "read"):
                        data.seek(0)
                        file.write(data.read())
                    else:
                        file.write(data)

                    file.close()

            await run_async(save_data_to_file)

        return f"{self.scheme}{full_file_path}"

    @asynccontextmanager
    async def open(self, file_path: str, mode: str = "r"):
        """
        Retrieve data from a specified file path, returning the content as bytes.

        This method opens the file in read mode and reads its content. The function expects the
        file to exist; if it does not, a FileNotFoundError will be raised.

        Parameters:
        -----------

            - file_path (str): The relative path of the file to retrieve data from.
            - mode (str): The mode to open the file, with "r" as the default for reading binary
              files. (default "r")

        Returns:
        --------

            The content of the retrieved file as bytes.
        """
        full_file_path = self._get_full_path(file_path)

        def get_file():
            return self.fs.open(full_file_path, mode=mode)

        file = await run_async(get_file)
        file = FileBufferedReader(file, name=f"{self.scheme}{full_file_path}")

        try:
            yield file
        finally:
            file.close()

    async def file_exists(self, file_path: str):
        """
        Check if a specified file exists in the filesystem.

        Parameters:
        -----------

            - file_path (str): The path of the file to check for existence.

        Returns:
        --------

            - bool: True if the file exists, otherwise False.
        """
        return await run_async(self.fs.exists, self._get_full_path(file_path))

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
        return await run_async(self.fs.isfile, self._get_full_path(file_path))

    async def get_size(self, file_path: str) -> int:
        return await run_async(
            self.s3.size, os.path.join(self.storage_path.replace("s3://", ""), file_path)
        )

    async def ensure_directory_exists(self, directory_path: str = ""):
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """
        if not directory_path.strip():
            directory_path = self.storage_path.replace(self.scheme, "")

        def ensure_directory():
            if not self.fs.exists(directory_path):
                self.fs.makedirs(directory_path, exist_ok=True)

        await run_async(ensure_directory)

    async def copy_file(self, source_file_path: str, destination_file_path: str):
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

        def copy():
            return self.fs.copy(
                self._get_full_path(source_file_path),
                self._get_full_path(destination_file_path),
                recursive=True,
            )

        return await run_async(copy)

    async def remove(self, file_path: str):
        """
        Remove the specified file from the filesystem if it exists.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        full_file_path = self._get_full_path(file_path)

        def remove_file():
            if self.fs.exists(full_file_path):
                self.fs.rm_file(full_file_path)

        await run_async(remove_file)

    async def remove_all(self, tree_path: Optional[str] = None):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        if tree_path is None:
            tree_path = self.storage_path.replace(self.scheme, "")
        else:
            tree_path = self._get_full_path(tree_path)

        # async_remove_all = run_async(lambda: self.s3.rm(tree_path, recursive=True))

        try:
            # await async_remove_all()
            await run_async(self.fs.rm, tree_path, recursive=True)
        except FileNotFoundError:
            pass
