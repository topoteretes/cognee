import os
import shutil
from typing import BinaryIO, Union
from .StorageManager import Storage


class LocalStorage(Storage):
    """
    Manage local file storage operations such as storing, retrieving, and managing files on
    the filesystem.
    """

    storage_path: str = None

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

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
        full_file_path = self.storage_path + "/" + file_path

        LocalStorage.ensure_directory_exists(self.storage_path)

        with open(
            full_file_path,
            mode="w" if isinstance(data, str) else "wb",
            encoding="utf-8" if isinstance(data, str) else None,
        ) as f:
            if hasattr(data, "read"):
                data.seek(0)
                f.write(data.read())
            else:
                f.write(data)

    def retrieve(self, file_path: str, mode: str = "rb"):
        """
        Retrieve data from a specified file path, returning the content as bytes.

        This method opens the file in read mode and reads its content. The function expects the
        file to exist; if it does not, a FileNotFoundError will be raised.

        Parameters:
        -----------

            - file_path (str): The relative path of the file to retrieve data from.
            - mode (str): The mode to open the file, with 'rb' as the default for reading binary
              files. (default 'rb')

        Returns:
        --------

            The content of the retrieved file as bytes.
        """
        full_file_path = self.storage_path + "/" + file_path

        with open(full_file_path, mode=mode) as f:
            f.seek(0)
            return f.read()

    @staticmethod
    def file_exists(file_path: str):
        """
        Check if a specified file exists in the filesystem.

        Parameters:
        -----------

            - file_path (str): The path of the file to check for existence.

        Returns:
        --------

            - bool: True if the file exists, otherwise False.
        """
        return os.path.exists(file_path)

    @staticmethod
    def ensure_directory_exists(file_path: str):
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - file_path (str): The path of the directory to check or create.
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

    @staticmethod
    def remove(file_path: str):
        """
        Remove the specified file from the filesystem if it exists.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def copy_file(source_file_path: str, destination_file_path: str):
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
        return shutil.copy2(source_file_path, destination_file_path)

    @staticmethod
    def remove_all(tree_path: str):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        try:
            shutil.rmtree(tree_path)
        except FileNotFoundError:
            pass
