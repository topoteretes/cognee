import os
from os import path
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage


@asynccontextmanager
async def open_data_file(file_path: str, mode: str = "rb", encoding: str = None, **kwargs):
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
        file_dir_path = path.dirname(actual_fs_path)
        file_name = path.basename(actual_fs_path)

        file_storage = LocalFileStorage(file_dir_path)

        with file_storage.open(file_name, mode=mode, encoding=encoding, **kwargs) as file:
            yield file

    elif file_path.startswith("s3://"):
        # Handle S3 URLs without normalization (which corrupts them)
        parsed_url = urlparse(file_path)

        normalized_url = (
            f"s3://{parsed_url.netloc}{os.sep}{os.path.normpath(parsed_url.path).lstrip(os.sep)}"
        )

        s3_dir_path = os.path.dirname(normalized_url)
        s3_filename = os.path.basename(normalized_url)

        # if "/" in s3_path:
        #     s3_dir = "/".join(s3_path.split("/")[:-1])
        #     s3_filename = s3_path.split("/")[-1]
        # else:
        #     s3_dir = ""
        #     s3_filename = s3_path

        # Extract filesystem path from S3 URL structure
        # file_dir_path = (
        #     f"s3://{parsed_url.netloc}/{s3_dir}" if s3_dir else f"s3://{parsed_url.netloc}"
        # )
        # file_name = s3_filename

        file_storage = S3FileStorage(s3_dir_path)

        async with file_storage.open(s3_filename, mode=mode, **kwargs) as file:
            yield file

    else:
        # Regular file path - normalize separators
        normalized_path = os.path.normpath(file_path)
        file_dir_path = path.dirname(normalized_path)
        file_name = path.basename(normalized_path)

        # Validate that we have a proper filename
        if not file_name or file_name == "." or file_name == "..":
            raise ValueError(f"Invalid filename extracted: '{file_name}' from path: '{file_path}'")

        file_storage = LocalFileStorage(file_dir_path)

        with file_storage.open(file_name, mode=mode, encoding=encoding, **kwargs) as file:
            yield file
