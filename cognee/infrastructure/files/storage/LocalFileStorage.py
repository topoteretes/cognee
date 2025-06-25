import os
import shutil
import asyncio
import aiofiles
import aiofiles.os
from aiofiles import ospath
from typing import BinaryIO, Optional, Union
from contextlib import asynccontextmanager

from .FileBufferedReader import FileBufferedReader
from .storage import Storage


@asynccontextmanager
async def async_open(path, *args, **kwargs):
    file = await asyncio.to_thread(open, path, *args, **kwargs)
    try:
        yield file
    finally:
        await asyncio.to_thread(file.close)


class LocalFileStorage(Storage):
    """
    Manage local file storage operations such as storing, retrieving, and managing files on
    the filesystem.
    """

    storage_path: Optional[str] = None

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

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
        full_file_path = os.path.join(self.storage_path.replace("file://", ""), file_path)
        file_dir_path = os.path.dirname(full_file_path)

        await self.ensure_directory_exists(file_dir_path)

        if overwrite or not await self.file_exists(file_path):
            async with aiofiles.open(
                full_file_path,
                mode="w" if isinstance(data, str) else "wb",
                encoding="utf-8" if isinstance(data, str) else None,
            ) as file:
                if hasattr(data, "read"):
                    await data.seek(0)
                    await file.write(data.read())
                else:
                    await file.write(data)

                await file.close()

        return "file://" + full_file_path

    @asynccontextmanager
    async def open(self, file_path: str, mode: str = "rb", *args, **kwargs):
        """
        Retrieve data from a specified file path, returning the content as bytes.

        This method opens the file in read mode and reads its content. The function expects the
        file to exist; if it does not, a FileNotFoundError will be raised.

        Parameters:
        -----------

            - file_path (str): The relative path of the file to retrieve data from.
            - mode (str): The mode to open the file, with "rb" as the default for reading binary
              files. (default "rb")

        Returns:
        --------

            The content of the retrieved file as bytes.
        """
        full_file_path = os.path.join(self.storage_path.replace("file://", ""), file_path)

        async with async_open(full_file_path, mode=mode, *args, **kwargs) as file:
            yield FileBufferedReader(file, name="file://" + full_file_path)

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
        return await ospath.exists(
            os.path.join(self.storage_path.replace("file://", ""), file_path)
        )

    async def ensure_directory_exists(self, directory_path: str = None):
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """
        if directory_path is None:
            directory_path = self.storage_path.replace("file://", "")

        if not await ospath.exists(directory_path):
            await aiofiles.os.makedirs(directory_path, exist_ok=True)

    def copy_file(self, source_file_path: str, destination_file_path: str):
        """
        Copy a file from a source path to a destination path.
        Files need to be in the same storage.

        Parameters:
        -----------

            - source_file_path (str): The path of the file to be copied.
            - destination_file_path (str): The path where the file will be copied to.

        Returns:
        --------

            - str: The path to the copied file.
        """
        return shutil.copy2(
            os.path.join(self.storage_path.replace("file://", ""), source_file_path),
            os.path.join(self.storage_path.replace("file://", ""), destination_file_path),
        )

    async def remove(self, file_path: str):
        """
        Remove the specified file from the storage if it exists.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        full_file_path = os.path.join(self.storage_path.replace("file://", ""), file_path)

        if await ospath.exists(full_file_path):
            await aiofiles.os.remove(full_file_path)

    async def remove_all(self, tree_path: str = None):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        If directories don't exist in the storage we ignore it.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        if tree_path is None:
            tree_path = self.storage_path.replace("file://", "")
        else:
            tree_path = os.path.join(self.storage_path.replace("file://", ""), tree_path)

        try:
            await asyncio.to_thread(shutil.rmtree, tree_path)
        except FileNotFoundError:
            pass
