from typing import BinaryIO, TypedDict
import hashlib
from .guess_file_type import guess_file_type
from cognee.shared.utils import get_file_content_hash


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


def get_file_metadata(file: BinaryIO) -> FileMetadata:
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
    file.seek(0)
    content_hash = get_file_content_hash(file)
    file.seek(0)

    file_type = guess_file_type(file)

    file_path = getattr(file, "name", None) or getattr(file, "full_name", None)
    file_name = str(file_path).split("/")[-1].split(".")[0] if file_path else None

    return FileMetadata(
        name=file_name,
        file_path=file_path,
        mime_type=file_type.mime,
        extension=file_type.extension,
        content_hash=content_hash,
    )
