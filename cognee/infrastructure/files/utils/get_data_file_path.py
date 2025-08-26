import os
from urllib.parse import urlparse


def get_data_file_path(file_path: str):
    # Check if this is a file URI BEFORE normalizing (which corrupts URIs)
    if file_path.startswith("file://"):
        # Remove first occurrence of file:// prefix
        pure_file_path = file_path.replace("file://", "", 1)
        # Normalize the file URI for Windows - replace backslashes with forward slashes
        normalized_file_uri = os.path.normpath(pure_file_path)

        # Convert path to proper file system path
        if os.name == "nt":  # Windows
            # Handle Windows drive letters correctly
            fs_path = normalized_file_uri
            if (
                (fs_path.startswith("/") or fs_path.startswith("\\"))
                and len(fs_path) > 1
                and fs_path[2] == ":"
            ):
                fs_path = fs_path[1:]
        else:
            # Unix - like systems
            fs_path = normalized_file_uri

        # Now split the actual filesystem path
        actual_fs_path = os.path.normpath(fs_path)
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
