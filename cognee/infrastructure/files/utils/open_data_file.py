import os
from os import path
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from cognee.infrastructure.files.storage import get_file_storage


@asynccontextmanager
async def open_data_file(file_path: str, mode: str = "rb", encoding: str = None, **kwargs):
    # Debug: Log the original file_path
    print(f"DEBUG: open_data_file called with file_path='{file_path}'")

    # Check if this is a file URI BEFORE normalizing (which corrupts URIs)
    if file_path.startswith("file://"):
        parsed_url = urlparse(file_path)

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
        file_dir_path = path.dirname(actual_fs_path)
        file_name = path.basename(actual_fs_path)

        print(
            f"DEBUG: file URI - actual_fs_path='{actual_fs_path}', file_dir_path='{file_dir_path}', file_name='{file_name}'"
        )

    elif file_path.startswith("s3://"):
        # Handle S3 URLs without normalization (which corrupts them)
        parsed_url = urlparse(file_path)

        # For S3, reconstruct the directory path and filename
        s3_path = parsed_url.path.lstrip("/")  # Remove leading slash

        if "/" in s3_path:
            s3_dir = "/".join(s3_path.split("/")[:-1])
            s3_filename = s3_path.split("/")[-1]
        else:
            s3_dir = ""
            s3_filename = s3_path

        # Extract filesystem path from S3 URL structure
        file_dir_path = (
            f"s3://{parsed_url.netloc}/{s3_dir}" if s3_dir else f"s3://{parsed_url.netloc}"
        )
        file_name = s3_filename

        print(
            f"DEBUG: S3 URL - s3_path='{s3_path}', file_dir_path='{file_dir_path}', file_name='{file_name}'"
        )

    else:
        # Regular file path - normalize separators
        normalized_path = os.path.normpath(file_path)
        file_dir_path = path.dirname(normalized_path)
        file_name = path.basename(normalized_path)

        print(
            f"DEBUG: regular path - normalized_path='{normalized_path}', file_dir_path='{file_dir_path}', file_name='{file_name}'"
        )

    # Validate that we have a proper filename
    if not file_name or file_name == "." or file_name == "..":
        raise ValueError(f"Invalid filename extracted: '{file_name}' from path: '{file_path}'")

    file_storage = get_file_storage(file_dir_path)

    async with file_storage.open(file_name, mode=mode, encoding=encoding, **kwargs) as file:
        yield file
