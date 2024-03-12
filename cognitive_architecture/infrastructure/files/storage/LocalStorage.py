import os
from typing import BinaryIO
from .StorageManager import Storage

class LocalStorage(Storage):
    storage_path: str = None

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def store(self, file_path: str, data: BinaryIO):
        full_file_path = self.storage_path + "/" + file_path

        LocalStorage.ensure_directory_exists(self.storage_path)

        with open(full_file_path, "wb") as f:
            f.write(data.read())

    def retrieve(self, file_path: str):
        full_file_path = self.storage_path + "/" + file_path

        with open(full_file_path, "rb") as f:
            return f.read()

    @staticmethod
    def ensure_directory_exists(file_path: str):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    def remove(self, file_path: str):
        os.remove(self.storage_path + "/" + file_path)

    # def get_directory(self, file_path: str):
    #     [path, __] = file_path.split(".")
    #     directory = "/".join(path.split("/")[:-1])

    #     return directory if directory != "" else None
