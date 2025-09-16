import os
from cognee.base_config import get_base_config
from .storage_provider_registry import StorageProviderRegistry
from .local_storage.LocalFileStorage import LocalFileStorage
from .StorageManager import StorageManager
from .utils import get_scheme_with_separator


def get_file_storage(storage_path: str) -> StorageManager:
    if not isinstance(storage_path, str) or not storage_path:
        raise ValueError(f"Invalid storage path: {storage_path}")

    try:
        base_config = get_base_config()
        all_cloud_schemes = StorageProviderRegistry.get_all_cloud_schemes()
        storage_path_scheme = get_scheme_with_separator(storage_path)
        system_root_directory_scheme = get_scheme_with_separator(base_config.system_root_directory)
        data_root_directory_scheme = get_scheme_with_separator(base_config.data_root_directory)

        # Use CloudFileStorage if the storage_path is a cloud storage URL or if configured for cloud storage
        if storage_path.startswith(all_cloud_schemes) or ():
            provider_cls = StorageProviderRegistry.get_provider_by_cloud_scheme(storage_path_scheme)
            return StorageManager(provider_cls(storage_path))
        elif (
            os.getenv("STORAGE_BACKEND") != "local"
            and system_root_directory_scheme == data_root_directory_scheme
            and system_root_directory_scheme.startswith(all_cloud_schemes)
            and data_root_directory_scheme.startswith(all_cloud_schemes)
        ):
            provider_cls = StorageProviderRegistry.get_provider_by_cloud_scheme(
                system_root_directory_scheme
            )
            return StorageManager(provider_cls(storage_path))

        else:
            return StorageManager(LocalFileStorage(storage_path))

    except Exception as exc:
        raise ValueError(f"Invalid storage path: {storage_path}") from exc
