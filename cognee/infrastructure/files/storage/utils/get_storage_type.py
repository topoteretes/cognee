import os
from urllib.parse import urlparse
from cognee.base_config import get_base_config


def get_storage_type(storage_path: str) -> str:
    if not isinstance(storage_path, str) or not storage_path:
        return "unknown"

    try:
        result = urlparse(storage_path)
        scheme = result.scheme.lower()
        base_config = get_base_config()

        # Use S3FileStorage if the storage_path is an S3 URL or if configured for S3
        if scheme in ("s3",) or (
            os.getenv("STORAGE_BACKEND") == "s3"
            and "s3://" in base_config.system_root_directory
            and "s3://" in base_config.data_root_directory
        ):
            return "s3"

        else:
            # other protocols, such as http, file, etc.
            return "local"
    except Exception:
        return "unknown (parsing_error)"
