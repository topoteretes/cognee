import hashlib
from dataclasses import dataclass
from typing import BinaryIO, Optional, Union

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config

# ``_derive_basename`` is the one place that turns a path/URI into the
# extension-less document name stored on ``Data.name``. Reusing it here keeps
# the name identical to the one the old read-it-back-from-storage path produced.
from cognee.infrastructure.files.utils.get_file_metadata import FileMetadata, _derive_basename

from .classify import classify
from .data_types import TextData


@dataclass
class StoredFile:
    """A payload written to cognee storage, plus the metadata describing it.

    ``metadata`` is computed from the bytes while they are still in hand, so no
    caller has to read the object back to learn its content hash, size, mime
    type or name. It is ``None`` only for items cognee did not write — an
    ``s3://`` URL or a local path handed straight through — where the bytes
    were never in this process to begin with.
    """

    file_path: str
    metadata: Optional[FileMetadata] = None


def _storage_key(file_name: str, content_hash: Optional[str], is_text: bool) -> str:
    """The object key a payload is stored under: ``<content-md5>/<filename>``.

    Keying on the caller's filename alone meant two different uploads that
    happened to share a name overwrote each other, because the write is an
    ``overwrite=True`` PUT. The content-hash prefix makes distinct payloads
    distinct and re-adding identical bytes idempotent.

    The basename stays the user's real filename on purpose: the code-graph
    route stages files under ``basename(original_data_location)`` and keys node
    identity on it (it must survive re-ingestion), loaders select by suffix,
    and dlt derives its source name from it. A flat ``<hash>.<ext>`` key breaks
    all three.

    Text payloads are left alone — they already have a content-addressed name
    (``text_<md5>.txt``) that other parts of cognee construct by hand and
    assert on, so it stays the single naming source for text.
    """
    if is_text or not content_hash:
        return file_name

    return f"{content_hash}/{file_name}"


async def save_data_to_file_detailed(
    data: Union[str, BinaryIO],
    filename: str = None,
    file_extension: Optional[str] = None,
) -> StoredFile:
    """Save ``data`` to cognee storage and describe what was saved.

    The returned ``metadata`` is built from the in-memory payload, which is the
    point: ingestion used to store the file and then download it again (several
    times) purely to recompute a hash of bytes it had just written.
    """
    storage_config = get_storage_config()

    data_root_directory = storage_config["data_root_directory"]

    classified_data = classify(data, filename)
    is_text = isinstance(classified_data, TextData)

    file_metadata = await classified_data.aget_metadata()

    async with classified_data.get_data() as payload:
        if "name" not in file_metadata or file_metadata["name"] is None:
            data_contents = payload.encode("utf-8")
            hash_contents = hashlib.md5(data_contents).hexdigest()
            file_metadata["name"] = "text_" + hash_contents + ".txt"

        file_name = file_metadata["name"]

        if file_extension is not None:
            extension = file_extension.lstrip(".")
            file_name_without_ext = file_name.rsplit(".", 1)[0]
            file_name = f"{file_name_without_ext}.{extension}"

        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(
            _storage_key(file_name, file_metadata.get("content_hash"), is_text),
            payload,
            overwrite=True,
        )

    # Point the metadata at where the payload actually landed, and carry the
    # document name derived from the caller's filename rather than from the
    # (now content-addressed) key — ``Data.name`` must stay the name a user
    # recognizes.
    file_metadata["file_path"] = full_file_path
    file_metadata["name"] = _derive_basename(file_name)

    return StoredFile(file_path=full_file_path, metadata=file_metadata)


async def save_data_to_file(
    data: Union[str, BinaryIO], filename: str = None, file_extension: Optional[str] = None
) -> str:
    """Save ``data`` to cognee storage and return its path.

    Thin wrapper over :func:`save_data_to_file_detailed` for callers that only
    need the path.
    """
    stored = await save_data_to_file_detailed(data, filename, file_extension)

    return stored.file_path
