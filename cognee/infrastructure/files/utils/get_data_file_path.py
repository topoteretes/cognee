import os
from urllib.parse import urlparse


def get_data_file_path(file_path: str):
    # Check if this is a file URI BEFORE normalizing (which corrupts URIs)
    if file_path.startswith("file://"):
        # Remove first occurrence of file:// prefix
        pure_file_path = file_path.replace("file://", "", 1)
        # Normalize the file URI for Windows - replace backslashes with forward slashes
        normalized_file_uri = os.path.normpath(pure_file_path)

        # Now split the actual filesystem path
        actual_fs_path = os.path.normpath(normalized_file_uri)
        return actual_fs_path

    elif file_path.startswith("s3://"):
        # Handle S3 URLs without normalization (which corrupts them)
        parsed_url = urlparse(file_path)

        normalized_url = (
            f"s3://{parsed_url.netloc}{os.sep}{os.path.normpath(parsed_url.path).lstrip(os.sep)}"
        )

        return normalized_url

    else:
        # Regular file path - normalize separators
        normalized_path = os.path.normpath(file_path)
        return normalized_path
