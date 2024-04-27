""" Config class is used to load the configuration from the config file. The config file is loaded in the get_graph_client function. The get_graph_client function is a factory function that returns the appropriate graph client based on the graph type. The graph_file_path is constructed using the database_directory_path and the graph_file_name. If the graph type is NEO4J, the function tries to import the Neo4jAdapter and return an instance of it. If the import fails, it falls back to using the NetworkXAdapter. The NetworkXAdapter is initialized with the graph_file_path, and if the graph is not loaded, it loads the graph from the file. The function returns the graph client instance. """
import os
import shutil
from typing import BinaryIO, Union
from .StorageManager import Storage

class LocalStorage(Storage):
    storage_path: str = None

    def __init__(self, storage_path: str) -> None:
        self.storage_path = storage_path

    def store(self, file_path: str, data: Union[BinaryIO, str]) -> None:
        full_file_path = self.storage_path + "/" + file_path

        LocalStorage.ensure_directory_exists(self.storage_path)

        with open(
            full_file_path,
            mode = "w" if isinstance(data, str) else "wb",
            encoding = "utf-8" if isinstance(data, str) else None
        ) as f:
            f.write(data if isinstance(data, str) else data.read())

    def retrieve(self, file_path: str, mode: str = "rb") -> bytes:
        full_file_path = self.storage_path + "/" + file_path

        with open(full_file_path, mode = mode) as f:
            return f.read()

    @staticmethod
    def ensure_directory_exists(file_path: str) -> None:
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    def remove(self, file_path: str) -> None:
        os.remove(self.storage_path + "/" + file_path)

    @staticmethod
    def copy_file(source_file_path: str, destination_file_path: str) -> str:
        return shutil.copy2(source_file_path, destination_file_path)

    @staticmethod
    def remove_all(tree_path: str) -> None:
        try:
            shutil.rmtree(tree_path)
        except FileNotFoundError:
            pass
