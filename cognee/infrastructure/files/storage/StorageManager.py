from typing import Protocol, BinaryIO


class Storage(Protocol):
    def store(self, file_path: str, data: bytes):
        pass

    def retrieve(self, file_path: str):
        pass

    @staticmethod
    def remove(file_path: str):
        pass


class StorageManager:
    storage: Storage = None

    def __init__(self, storage: Storage):
        self.storage = storage

    def store(self, file_path: str, data: BinaryIO):
        return self.storage.store(file_path, data)

    def retrieve(self, file_path: str):
        return self.storage.retrieve(file_path)

    def remove(self, file_path: str):
        return self.storage.remove(file_path)
