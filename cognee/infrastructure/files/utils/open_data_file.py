from nturl2path import url2pathname
import os
from os import path
from pathlib import Path
from urllib.parse import unquote, urlparse
from contextlib import asynccontextmanager

from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage


@asynccontextmanager
async def open_data_file(file_path: str, mode: str = "rb", encoding: str = None, **kwargs):
    # Check if this is a file URI BEFORE normalizing (which corrupts URIs)
    if file_path.startswith("file://"):
        # Use pathlib.Path.from_uri() when bump Python to 3.13
        # See https://github.com/python/cpython/issues/107465
        p = urlparse(file_path)
        raw = unquote(p.path)

        # Windows: file:///C:/dir/file.txt -> "/C:/dir/file.txt" -> "C:/dir/file.txt"
        if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]

        fs_path = Path(raw)

        file_storage = LocalFileStorage(str(fs_path.parent))
        with file_storage.open(fs_path.name, mode=mode, encoding=encoding, **kwargs) as f:
            yield f

    elif file_path.startswith("s3://"):
        normalized_url = get_data_file_path(file_path)
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
        fs_path = Path(file_path).resolve(strict=False)

        if not fs_path.name or fs_path.name in (".", ".."):
            raise ValueError(
                f"Invalid filename extracted: '{fs_path.name}' from path: '{file_path}'"
            )

        file_storage = LocalFileStorage(str(fs_path.parent))
        with file_storage.open(fs_path.name, mode=mode, encoding=encoding, **kwargs) as file:
            yield file
