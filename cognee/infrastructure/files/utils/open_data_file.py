import os
from os import path
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from cognee.infrastructure.files.storage import get_file_storage


@asynccontextmanager
async def open_data_file(file_path: str, mode: str = "rb", encoding: str = None, **kwargs):
    # Check if this is a file URI BEFORE normalizing (which corrupts URIs)
    if file_path.startswith("file://"):
        parsed_url = urlparse(file_path)

        # Convert URI path to file system path
        if os.name == "nt":  # Windows
            # Handle Windows drive letters correctly
            fs_path = parsed_url.path
            if fs_path.startswith("/") and len(fs_path) > 1 and fs_path[2] == ":":
                fs_path = fs_path[1:]  # Remove leading slash for drive letters like /C:/
        else:  # Unix-like systems
            fs_path = parsed_url.path

        # Normalize the extracted file system path
        fs_path = os.path.normpath(fs_path)

        file_dir_path = path.dirname(fs_path)
        file_name = path.basename(fs_path)
    else:
        # Regular file path - normalize and split
        normalized_path = os.path.normpath(file_path)
        file_dir_path = path.dirname(normalized_path)
        file_name = path.basename(normalized_path)

    file_storage = get_file_storage(file_dir_path)

    async with file_storage.open(file_name, mode=mode, encoding=encoding, **kwargs) as file:
        yield file
