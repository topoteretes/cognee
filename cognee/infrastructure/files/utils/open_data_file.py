import os
import posixpath
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from os import path

from cognee.infrastructure.files.encryption import wrap_storage_if_encrypted
from cognee.infrastructure.files.storage.FileBufferedReader import FileBufferedReader
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.infrastructure.files.storage.remote_storage import (
    get_remote_scheme,
    get_remote_storage_class,
)
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path


def _with_encoding(kwargs: dict, encoding: str | None) -> dict:
    """Keyword arguments for a local open(), with ``encoding`` only when the caller set it.

    The local backend forwards extra keyword arguments to the builtin ``open()``,
    where an explicit ``None`` equals omitting it; the Storage protocol itself does
    not name ``encoding``, and the object-store backends' open() takes none at all,
    so the remote branch never receives it.
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

    elif (scheme := get_remote_scheme(file_path)) is not None:
        # An object-store URL (s3://, az://): read through the backend for its scheme,
        # which raises its own install hint when its extra is missing.
        normalized_url = get_data_file_path(file_path)
        storage_class = get_remote_storage_class(scheme)

        file_dir_path, file_name = posixpath.split(normalized_url)
        file_storage = wrap_storage_if_encrypted(storage_class(file_dir_path))

        async with file_storage.open(file_name, mode=mode, **kwargs) as file:
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
