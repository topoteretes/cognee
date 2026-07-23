import io
import os.path
from pathlib import PureWindowsPath
from typing import BinaryIO, Optional, TypedDict
from urllib.parse import unquote, urlparse

from cognee.infrastructure.files.utils.get_file_content_hash import get_file_content_hash
from cognee.shared.logging_utils import get_logger

from .guess_file_type import guess_file_type

logger = get_logger("FileMetadata")


def _derive_basename(file_path: str) -> Optional[str]:
    """Derive a clean, extension-less document name from a path or file:// URI.

    In the ingestion pipeline ``file.name`` is a percent-encoded ``file://`` URI
    (LocalFileStorage wraps every opened file in a FileBufferedReader whose name is
    ``Path(full_path).as_uri()``), but other callers may pass a raw filesystem path.
    ``Path(file_path).stem`` alone mishandles both: it leaves ``%20`` and other
    percent-escapes in the name, and it is not OS-agnostic (POSIX ``Path`` does not
    treat "\\" as a separator, so a Windows-style path yields the whole path as the
    "stem", and vice versa). This normalizes both cases:

    * percent-decodes ``file://`` URIs (so "Report%20Q1.pdf" -> "Report Q1"),
    * resolves the basename + stem with ``PureWindowsPath``, which treats both "/"
      and "\\" as separators on every host OS (so it is OS-agnostic),
    * strips a single trailing extension, preserving the prior ``Path(...).stem``
      semantics (the extension is stored separately in ``FileMetadata["extension"]``).
    """
    candidate = file_path
    if candidate.startswith("file://"):
        candidate = unquote(urlparse(candidate).path)

    # ``PureWindowsPath`` treats both "/" and "\\" as separators regardless of host
    # OS, so basename + stem resolution is OS-independent (a plain ``Path`` /
    # ``PurePosixPath`` would not split "\\"). Percent-decoding above is URL work
    # that pathlib cannot do.
    return PureWindowsPath(candidate).stem or None


class FileMetadata(TypedDict):
    """
    Represents metadata for a file.

    This class defines a structure to store various attributes related to a file, including
    its name, file path, MIME type, file extension, and a content hash for integrity
    checking.
    """

    name: str | None
    file_path: str | None
    mime_type: str
    extension: str
    content_hash: str
    file_size: int


async def get_file_metadata(file: BinaryIO, name: str | None = None) -> FileMetadata:
    """
    Retrieve metadata from a file object.

    Reset the file pointer to the beginning of the file and compute the content hash. Guess
    the file type and extract the file path and name. Construct and return a dictionary
    containing the file's metadata attributes.

    Parameters:
    -----------

        - file (BinaryIO): A file-like object from which to extract metadata.

    Returns:
    --------

        - FileMetadata: A dictionary containing the file's name, path, MIME type, file
          extension, and content hash.
    """
    content_hash = ""
    try:
        file.seek(0)
        content_hash = await get_file_content_hash(file)
        file.seek(0)
    except io.UnsupportedOperation as error:
        logger.error(f"Error retrieving content hash for file: {file.name} \n{str(error)}\n\n")

    file_type = guess_file_type(file, name)

    file_path = getattr(file, "name", None) or getattr(file, "full_name", None)

    if isinstance(file_path, str):
        file_name = _derive_basename(file_path)
    else:
        # In case file_path does not exist or is a integer return None
        file_name = None

    # Get file size
    pos = file.tell()  # remember current pointer
    file.seek(0, os.SEEK_END)  # jump to end
    file_size = file.tell()  # byte count
    file.seek(pos)

    return FileMetadata(
        name=file_name,
        file_path=file_path,
        mime_type=file_type.mime,
        extension=file_type.extension,
        content_hash=content_hash,
        file_size=file_size,
    )
