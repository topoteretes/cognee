import os

from .StorageManager import StorageManager


def get_file_storage(storage_path: str) -> StorageManager:
    if os.getenv("STORAGE_BACKEND") == "s3":
        from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

        return StorageManager(S3FileStorage(storage_path))
    else:
        from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage

        return StorageManager(LocalFileStorage(storage_path))
