from typing import BinaryIO, Protocol, Union


class Storage(Protocol):
    storage_path: str

    """
    Abstract interface for storage operations.
    """

    def file_exists(self, file_path: str) -> bool:
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

    def is_file(self, file_path: str) -> bool:
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

    def get_size(self, file_path: str) -> int:
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

    def store(self, file_path: str, data: Union[BinaryIO, str], overwrite: bool):
        """
        Store data at the specified file path.

        Parameters:
        -----------

            - file_path (str): The path where the data will be stored.
            - data (bytes): The binary data to be stored.
            - overwrite (bool): If True, overwrite the existing file.
        """
        pass

    def open(self, file_path: str, mode: str = "r"):
        """
        Retrieve file from the specified file path.

        Parameters:
        -----------

            - file_path (str): The path from where the data will be retrieved.
            - mode (str): The mode to open the file, with "r" as the default for reading text
        """
        pass

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
        pass

    def ensure_directory_exists(self, directory_path: str = ""):
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """
        pass

    def remove(self, file_path: str):
        """
        Remove the storage at the specified file path.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        pass

    def remove_all(self, root_path: str = None):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        pass
