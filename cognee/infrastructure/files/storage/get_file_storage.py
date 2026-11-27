from cognee.base_config import get_base_config

from .remote_storage import get_remote_scheme, get_remote_storage_class, is_configured_backend
from .StorageManager import StorageManager


def get_file_storage(storage_path: str) -> StorageManager:
    base_config = get_base_config()

    # Imported here: the encryption package reads storage modules of this package.
    from cognee.infrastructure.files.encryption import wrap_storage_if_encrypted

    # A remote URL selects its object store, and so does a configured remote backend
    # whose roots are both on it (S3 or Azure Blob).
    scheme = get_remote_scheme(storage_path) or _configured_remote_scheme(base_config)
    if scheme is not None:
        storage = get_remote_storage_class(scheme)(storage_path)
    else:
        from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage

        storage = LocalFileStorage(storage_path)

    # Roots under DATA_ROOT_DIRECTORY are sealed when FILE_ENCRYPTION_ENABLED
    # is on; every other root (system, cache) is handed back as is.
    return StorageManager(wrap_storage_if_encrypted(storage))


def _configured_remote_scheme(base_config) -> str | None:
    """The remote scheme STORAGE_BACKEND names, when the system and data roots are both on it."""
    scheme = get_remote_scheme(base_config.system_root_directory)
    if scheme is None or scheme != get_remote_scheme(base_config.data_root_directory):
        return None
    return scheme if is_configured_backend(scheme) else None
