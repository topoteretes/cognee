import os
import shutil
from typing import BinaryIO, Union
from .StorageManager import Storage


class LocalStorage(Storage):
    storage_path: str = None

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def store(self, file_path: str, data: Union[BinaryIO, str]):
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
        full_file_path = self.storage_path + "/" + file_path

        with open(full_file_path, mode=mode) as f:
            f.seek(0)
            return f.read()

    @staticmethod
    def file_exists(file_path: str):
        return os.path.exists(file_path)

    @staticmethod
    def ensure_directory_exists(file_path: str):
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

    @staticmethod
    def remove(file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def copy_file(source_file_path: str, destination_file_path: str):
        return shutil.copy2(source_file_path, destination_file_path)

    @staticmethod
    def remove_all(tree_path: str):
        try:
            shutil.rmtree(tree_path)
        except FileNotFoundError:
            pass
