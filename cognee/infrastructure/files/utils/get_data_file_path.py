import os
from urllib.parse import urlparse
from cognee.infrastructure.files.storage import StorageProviderRegistry


def get_data_file_path(file_path: str):
    # Check if this is a file URI BEFORE normalizing (which corrupts URIs)
    if file_path.startswith("file://"):
        # Normalize the file URI for Windows - replace backslashes with forward slashes
        normalized_file_uri = os.path.normpath(file_path)

        parsed_url = urlparse(normalized_file_uri)

        # Convert URI path to file system path
        if os.name == "nt":  # Windows
            # Handle Windows drive letters correctly
            fs_path = parsed_url.path
            if fs_path.startswith("/") and len(fs_path) > 1 and fs_path[2] == ":":
                fs_path = fs_path[1:]  # Remove leading slash for Windows drive paths
        else:  # Unix-like systems
            fs_path = parsed_url.path

        # Now split the actual filesystem path
        actual_fs_path = os.path.normpath(fs_path)
        return actual_fs_path

    elif file_path.startswith(StorageProviderRegistry.get_all_cloud_schemes()):
        # Handle cloud storage(s3, gcs, azure, etc.) URLs without normalization (which corrupts them)
        parsed_url = urlparse(file_path)

        normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{os.sep}{os.path.normpath(parsed_url.path).lstrip(os.sep)}"

        return normalized_url

    else:
        # Regular file path - normalize separators
        normalized_path = os.path.normpath(file_path)
        return normalized_path
