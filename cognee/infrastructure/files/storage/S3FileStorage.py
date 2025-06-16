import os
import s3fs
from typing import BinaryIO, Union
from contextlib import contextmanager
from .StorageManager import Storage


class S3FileStorage(Storage):
    """
    Manage local file storage operations such as storing, retrieving, and managing files on
    the filesystem.
    """

    storage_path: str
    s3: s3fs.S3FileSystem

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.s3 = s3fs.S3FileSystem(anon=True)

    def store(self, file_path: str, data: Union[BinaryIO, str]):
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
        """
        full_file_path = os.path.join(self.storage_path, file_path)

        file_dir_path = os.path.dirname(full_file_path)

        self.ensure_directory_exists(file_dir_path)

        with self.s3.open(
            full_file_path,
            mode="w" if isinstance(data, str) else "wb",
            encoding="utf-8" if isinstance(data, str) else None,
        ) as file:
            if hasattr(data, "read"):
                data.seek(0)
                file.write(data.read())
            else:
                file.write(data)

    @contextmanager
    def open(self, file_path: str, mode: str = "r"):
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
        full_file_path = os.path.join(self.storage_path, file_path)

        with self.s3.open(full_file_path, mode=mode) as file:
            yield file

    def file_exists(self, file_path: str):
        """
        Check if a specified file exists in the filesystem.

        Parameters:
        -----------

            - file_path (str): The path of the file to check for existence.

        Returns:
        --------

            - bool: True if the file exists, otherwise False.
        """
        return self.s3.exists(os.path.join(self.storage_path, file_path))

    def ensure_directory_exists(self, file_path: str):
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - file_path (str): The path of the directory to check or create.
        """
        directory_path = os.path.dirname(directory_path)

        if not self.file_exists(directory_path):
            self.s3.makedirs(directory_path, exist_ok=True)

    def copy_file(self, source_file_path: str, destination_file_path: str):
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
        return self.s3.copy(
            os.path.join(self.storage_path, source_file_path),
            os.path.join(self.storage_path, destination_file_path),
            recursive=True,
        )

    def remove(self, file_path: str):
        """
        Remove the specified file from the filesystem if it exists.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        full_file_path = os.path.join(self.storage_path, file_path)

        if self.file_exists(full_file_path):
            self.s3.rm_file(full_file_path)

    def remove_all(self, tree_path: str):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        try:
            self.s3.rm(os.path.join(self.storage_path, tree_path), recursive=True)
        except FileNotFoundError:
            pass
