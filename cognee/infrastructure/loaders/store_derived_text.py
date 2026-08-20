"""Storing a loader's extracted text, described without reading it back."""

import hashlib
import io
import os

from cognee.infrastructure.files.utils.get_file_metadata import FileMetadata, _derive_basename
from cognee.infrastructure.files.utils.guess_file_type import guess_file_type

from .LoaderInterface import LoaderResult


def describe_derived_text(content: str, file_path: str) -> FileMetadata:
    """Describe extracted text exactly as reading the stored file back would.

    Both storage backends write a ``str`` payload as UTF-8 with ``newline="\\n"``
    (no translation), so the bytes hashed here are the bytes on disk. The type
    guess gets the stored file's name, exactly as the read-back path passed it
    (the opened file's name) — the ``.txt`` name is what routes derived text to
    ``text/plain``. Guessing from bytes alone misroutes any text that happens to
    start with another format's magic ("BM…", "%PDF…", "ID3…").
    """
    data_contents = content.encode("utf-8")
    file_type = guess_file_type(io.BytesIO(data_contents), os.path.basename(file_path))

    return FileMetadata(
        name=_derive_basename(file_path),
        file_path=file_path,
        mime_type=file_type.mime,
        extension=file_type.extension,
        # md5 hexdigest: the same digest get_file_content_hash produces, so
        # hashes already persisted on Data rows stay comparable.
        content_hash=hashlib.md5(data_contents).hexdigest(),
        file_size=len(data_contents),
    )


async def store_derived_text(
    storage, storage_file_name: str, content: str, **kwargs
) -> LoaderResult:
    """Store a loader's extracted text and return it already described.

    Ingestion needs the derived file's hash, size and type to build the ``Data``
    row. It used to get them by re-opening the file the loader had just written —
    over S3 that is a HEAD plus a full GET of content still in memory here.
    ``kwargs`` are forwarded to ``LoaderResult`` for loaders that also own the
    record's identity or route stamp.
    """
    file_path = await storage.store(storage_file_name, content)

    return LoaderResult(
        file_path=file_path,
        file_metadata=describe_derived_text(content, file_path),
        **kwargs,
    )
