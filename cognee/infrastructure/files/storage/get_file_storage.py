import os
from urllib.parse import urlparse
from cognee.base_config import get_base_config
from .storage_provider_registry import StorageProviderRegistry
from .local_storage.LocalFileStorage import LocalFileStorage
from .StorageManager import StorageManager


def get_file_storage(storage_path: str) -> StorageManager:
    if not isinstance(storage_path, str) or not storage_path:
        raise ValueError(f"Invalid storage path: {storage_path}")

    try:
        result = urlparse(storage_path)
        scheme = result.scheme.lower()
        scheme_with_separator = f"{scheme}://"
        base_config = get_base_config()

        # Use CloudFileStorage if the storage_path is a cloud storage URL or if configured for cloud storage
        if scheme_with_separator in StorageProviderRegistry.get_all_cloud_schemes() or (
            os.getenv("STORAGE_BACKEND")
            == StorageProviderRegistry.get_name_by_cloud_scheme(scheme_with_separator)
            and scheme_with_separator in base_config.system_root_directory
            and scheme_with_separator in base_config.data_root_directory
        ):
            provider_cls = StorageProviderRegistry.get_provider_by_cloud_scheme(
                scheme_with_separator
            )
            return StorageManager(provider_cls(storage_path))
        else:
            return StorageManager(LocalFileStorage(storage_path))

    except Exception as exc:
        raise ValueError(f"Invalid storage path: {storage_path}") from exc
