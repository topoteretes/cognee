from typing import Protocol, BinaryIO


class Storage(Protocol):
    """
    Abstract interface for storage operations.
    """

    def store(self, file_path: str, data: bytes):
        """
        Store data at the specified file path.

        Parameters:
        -----------

            - file_path (str): The path where the data will be stored.
            - data (bytes): The binary data to be stored.
        """
        pass

    def retrieve(self, file_path: str):
        """
        Retrieve data from the specified file path.

        Parameters:
        -----------

            - file_path (str): The path from where the data will be retrieved.
        """
        pass

    @staticmethod
    def remove(file_path: str):
        """
        Remove the storage at the specified file path.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        pass


class StorageManager:
    """
    Manages storage operations by delegating tasks to a storage backend.

    Public methods include:
    - store: Store data in the specified path.
    - retrieve: Retrieve data from the specified path.
    - remove: Remove the file at the specified path.
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

    def retrieve(self, file_path: str):
        """
        Retrieve data from the specified file path.

        Parameters:
        -----------

            - file_path (str): The path from which to retrieve the data.

        Returns:
        --------

            Returns the retrieved data, as defined by the storage implementation.
        """
        return self.storage.retrieve(file_path)

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
