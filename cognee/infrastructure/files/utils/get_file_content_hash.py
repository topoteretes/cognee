import hashlib
import os
from os import path
from typing import BinaryIO, Union

from ..exceptions import FileContentHashingError
from ..storage import get_file_storage
from .local_path_safety import resolve_local_path


# Bytes hashed per iteration. md5's block_size (64) was used here before,
# but that constant is the digest's internal compression block, not an I/O
# size — and FileBufferedReader forwards read() straight to the wrapped
# object, so every 64-byte call went through the whole storage stack
# (~16k Python-level calls per MiB). The digest is chunk-size independent.
HASH_CHUNK_SIZE = 1024 * 1024


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
                while chunk := file.read(HASH_CHUNK_SIZE):
                    h.update(chunk)
        else:
            while chunk := file_obj.read(HASH_CHUNK_SIZE):
                h.update(chunk)

        return h.hexdigest()
    except OSError as e:
        raise FileContentHashingError(message=f"Failed to hash data from {file_obj}: {e}")
