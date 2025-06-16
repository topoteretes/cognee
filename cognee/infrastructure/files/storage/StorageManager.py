from typing import Protocol, BinaryIO, Union


class Storage(Protocol):
    """
    Abstract interface for storage operations.
    """

    def store(self, file_path: str, data: Union[BinaryIO, str]):
        """
        Store data at the specified file path.

        Parameters:
        -----------

            - file_path (str): The path where the data will be stored.
            - data (bytes): The binary data to be stored.
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

    def remove(self, file_path: str):
        """
        Remove the storage at the specified file path.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        pass

    def remove_all(self, root_path: str):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        pass

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
        return self.storage.open(file_path, *args, **kwargs)

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

    def remove_all(self, tree_path: str):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        return self.storage.remove_all(tree_path)
