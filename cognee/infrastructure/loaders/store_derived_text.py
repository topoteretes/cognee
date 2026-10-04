"""Storing a loader's extracted text, described without reading it back."""

import io

from cognee.infrastructure.files.utils.get_file_metadata import (
    FileMetadata,
    _derive_basename,
    get_file_metadata,
)

from .LoaderInterface import LoaderResult


async def describe_derived_text(content: str, file_path: str) -> FileMetadata:
    """Describe extracted text exactly as reading the stored file back would.

    Both storage backends write a ``str`` payload as UTF-8 with ``newline="\\n"``
    (no translation), so the bytes described here are the bytes on disk — which
    is why this goes through :func:`get_file_metadata`, the same implementation
    the read-back path uses, over an in-memory view of those bytes. One source
    of truth for hashing, type detection, and size; only the name and path are
    filled in here, because a ``BytesIO`` carries neither.

    The name matters: the ``.txt`` name is what routes derived text to
    ``text/plain`` — guessing from bytes alone misroutes any text that happens
    to start with another format's magic ("BM…", "%PDF…", "ID3…").
    """
    data_contents = content.encode("utf-8")

    # ``name`` reaches guess_file_type exactly as the opened file's name would
    # on the read-back path.
    metadata = await get_file_metadata(io.BytesIO(data_contents), name=file_path)
    metadata["name"] = _derive_basename(file_path)
    metadata["file_path"] = file_path

    return metadata


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
        file_metadata=await describe_derived_text(content, file_path),
        **kwargs,
    )
