import os
import shutil
from urllib.parse import urlparse
from contextlib import contextmanager
from typing import BinaryIO, Optional, Union

from .FileBufferedReader import FileBufferedReader
from .storage import Storage


def get_parsed_path(file_path: str) -> str:
    parsed_url = urlparse(file_path)

    # On Windows, urlparse handles drive letters correctly
    # Convert the path component to a proper file path
    if os.name == "nt":  # Windows
        # Remove leading slash from Windows paths like /C:/Users/...
        parsed_path = parsed_url.path.lstrip("/")
    else:  # Unix-like systems
        parsed_path = parsed_url.path

    return parsed_path


class LocalFileStorage(Storage):
    """
    Manage local file storage operations such as storing, retrieving, and managing files on
    the filesystem.
    """

    storage_path: Optional[str] = None

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def store(self, file_path: str, data: Union[BinaryIO, str], overwrite: bool = False) -> str:
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
        parsed_storage_path = get_parsed_path(self.storage_path)
        full_file_path = os.path.join(parsed_storage_path, file_path)
        file_dir_path = os.path.dirname(full_file_path)

        self.ensure_directory_exists(file_dir_path)

        if overwrite or not os.path.exists(full_file_path):
            with open(
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

        return "file://" + full_file_path

    @contextmanager
    def open(self, file_path: str, mode: str = "rb", *args, **kwargs):
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
        parsed_storage_path = get_parsed_path(self.storage_path)

        full_file_path = os.path.join(parsed_storage_path, file_path)

        with open(full_file_path, mode=mode, *args, **kwargs) as file:
            file = FileBufferedReader(file, name="file://" + full_file_path)

            try:
                yield file
            finally:
                file.close()

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
        parsed_storage_path = get_parsed_path(self.storage_path)

        return os.path.exists(os.path.join(parsed_storage_path, file_path))

    def ensure_directory_exists(self, directory_path: str = None):
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """
        if directory_path is None:
            directory_path = get_parsed_path(self.storage_path)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

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
        parsed_storage_path = get_parsed_path(self.storage_path)

        return shutil.copy2(
            os.path.join(parsed_storage_path, source_file_path),
            os.path.join(parsed_storage_path, destination_file_path),
        )

    def remove(self, file_path: str):
        """
        Remove the specified file from the storage if it exists.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        parsed_storage_path = get_parsed_path(self.storage_path)
        full_file_path = os.path.join(parsed_storage_path, file_path)

        if os.path.exists(full_file_path):
            os.remove(full_file_path)

    def remove_all(self, tree_path: str = None):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        If directories don't exist in the storage we ignore it.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        parsed_storage_path = get_parsed_path(self.storage_path)

        if tree_path is None:
            tree_path = parsed_storage_path
        else:
            tree_path = os.path.join(parsed_storage_path, tree_path)

        try:
            return shutil.rmtree(tree_path)
        except FileNotFoundError:
            pass
