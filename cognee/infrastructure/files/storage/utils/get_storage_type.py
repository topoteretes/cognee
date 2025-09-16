from urllib.parse import urlparse
from ..storage_provider_registry import StorageProviderRegistry


def get_storage_type(storage_path: str) -> str:
    if not isinstance(storage_path, str) or not storage_path:
        raise ValueError(f"Invalid storage path: {storage_path}")

    try:
        result = urlparse(storage_path)
        scheme = result.scheme.lower()
        scheme_with_separator = f"{scheme}://"

        if scheme_with_separator in StorageProviderRegistry.get_all_cloud_schemes():
            return StorageProviderRegistry.get_name_by_cloud_scheme(scheme)
        else:
            return "local"

    except Exception as exc:
        raise ValueError(f"Invalid storage path: {storage_path}") from exc
