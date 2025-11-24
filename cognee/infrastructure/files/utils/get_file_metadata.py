import io
import os.path
from typing import BinaryIO, TypedDict, Optional
from pathlib import Path

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.files.utils.get_file_content_hash import get_file_content_hash
from .guess_file_type import guess_file_type

logger = get_logger("FileMetadata")


class FileMetadata(TypedDict):
    """
    Represents metadata for a file.

    This class defines a structure to store various attributes related to a file, including
    its name, file path, MIME type, file extension, and a content hash for integrity
    checking.
    """

    name: str
    file_path: str
    mime_type: str
    extension: str
    content_hash: str
    file_size: int


async def get_file_metadata(file: BinaryIO, name: Optional[str] = None) -> FileMetadata:
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
    try:
        file.seek(0)
        content_hash = await get_file_content_hash(file)
        file.seek(0)
    except io.UnsupportedOperation as error:
        logger.error(f"Error retrieving content hash for file: {file.name} \n{str(error)}\n\n")

    file_type = guess_file_type(file, name)

    file_path = getattr(file, "name", None) or getattr(file, "full_name", None)

    if isinstance(file_path, str):
        file_name = Path(file_path).stem if file_path else None
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
