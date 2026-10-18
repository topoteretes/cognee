import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from os import path
from urllib.parse import urlparse

from cognee.infrastructure.files.encryption import wrap_storage_if_encrypted
from cognee.infrastructure.files.storage.FileBufferedReader import FileBufferedReader
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path


def _with_encoding(kwargs: dict, encoding: str | None) -> dict:
    """Keyword arguments for a local open(), with ``encoding`` only when the caller set it.

    The local backend forwards extra keyword arguments to the builtin ``open()``,
    where an explicit ``None`` equals omitting it; the Storage protocol itself does
    not name ``encoding``, and S3FileStorage.open() takes none at all, so the S3
    branch never receives it.
    """
    if encoding is None:
        return kwargs
    return {**kwargs, "encoding": encoding}


@asynccontextmanager
async def open_data_file(
    file_path: str, mode: str = "rb", encoding: str | None = None, **kwargs
) -> AsyncGenerator[FileBufferedReader]:
    # Check if this is a file URI BEFORE normalizing (which corrupts URIs)
    if file_path.startswith("file://"):
        # Now split the actual filesystem path. Local reads go through the same
        # allowlist as ingestion (cognee's own storage roots are always allowed);
        # a path outside the allowed roots is reported as not found.
        try:
            actual_fs_path = os.fspath(resolve_local_path(get_data_file_path(file_path)))
        except ValueError as error:
            raise FileNotFoundError(file_path) from error
        file_dir_path = path.dirname(actual_fs_path)
        file_name = path.basename(actual_fs_path)

        # Sealed when the directory is under the data root and encryption is on;
        # a plain user path, or a plaintext object, comes back unchanged.
        file_storage = wrap_storage_if_encrypted(LocalFileStorage(file_dir_path))

        async with file_storage.open(
            file_name, mode=mode, **_with_encoding(kwargs, encoding)
        ) as file:
            yield file

    elif file_path.startswith("s3://"):
        try:
            from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage
        except ImportError:
            raise ImportError(
                "S3 dependencies are not installed. Please install with 'pip install cognee\"[aws]\"' to use S3 functionality."
            )

        normalized_url = get_data_file_path(file_path)
        s3_dir_path = os.path.dirname(normalized_url)
        s3_filename = os.path.basename(normalized_url)

        file_storage = wrap_storage_if_encrypted(S3FileStorage(s3_dir_path))

        async with file_storage.open(s3_filename, mode=mode, **kwargs) as file:
            yield file

    else:
        # Regular file path - normalize separators. Local reads go through the
        # same allowlist as ingestion; a path outside the allowed roots is
        # reported as not found.
        try:
            normalized_path = os.fspath(resolve_local_path(get_data_file_path(file_path)))
        except ValueError as error:
            raise FileNotFoundError(file_path) from error
        file_dir_path = path.dirname(normalized_path)
        file_name = path.basename(normalized_path)

        # Validate that we have a proper filename
        if not file_name or file_name == "." or file_name == "..":
            raise ValueError(f"Invalid filename extracted: '{file_name}' from path: '{file_path}'")

        # Sealed when the directory is under the data root and encryption is on;
        # a plain user path, or a plaintext object, comes back unchanged.
        file_storage = wrap_storage_if_encrypted(LocalFileStorage(file_dir_path))

        async with file_storage.open(
            file_name, mode=mode, **_with_encoding(kwargs, encoding)
        ) as file:
            yield file
