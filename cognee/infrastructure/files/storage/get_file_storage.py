import os

from cognee.base_config import get_base_config

from .StorageManager import StorageManager


def get_file_storage(storage_path: str) -> StorageManager:
    base_config = get_base_config()

    # Use S3FileStorage if the storage_path is an S3 URL or if configured for S3
    if storage_path.startswith("s3://") or (
        os.getenv("STORAGE_BACKEND") == "s3"
        and "s3://" in base_config.system_root_directory
        and "s3://" in base_config.data_root_directory
    ):
        from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

        return StorageManager(S3FileStorage(storage_path))
    else:
        from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage

        return StorageManager(LocalFileStorage(storage_path))
