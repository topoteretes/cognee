import hashlib
import os
from os import path
from typing import BinaryIO, Union

from ..exceptions import FileContentHashingError
from ..storage import get_file_storage
from .local_path_safety import resolve_local_path


async def get_file_content_hash(file_obj: str | BinaryIO) -> str:
    h = hashlib.md5()

    try:
        if isinstance(file_obj, str):
            # Normalize path separators (mixed separators on Windows) and go
            # through the same local-file allowlist as ingestion.
            try:
                normalized_path = os.fspath(resolve_local_path(file_obj))
            except ValueError as error:
                raise FileContentHashingError(
                    message=f"Failed to hash data from {file_obj}: path outside allowed roots."
                ) from error

            file_dir_path = path.dirname(normalized_path)
            file_name = path.basename(normalized_path)

            file_storage = get_file_storage(file_dir_path)

            async with file_storage.open(file_name, "rb") as file:
                while True:
                    # Reading is buffered, so we can read smaller chunks.
                    chunk = file.read(h.block_size)
                    if not chunk:
                        break
                    h.update(chunk)
        else:
            while True:
                # Reading is buffered, so we can read smaller chunks.
                chunk = file_obj.read(h.block_size)
                if not chunk:
                    break
                h.update(chunk)

        return h.hexdigest()
    except OSError as e:
        raise FileContentHashingError(message=f"Failed to hash data from {file_obj}: {e}")
