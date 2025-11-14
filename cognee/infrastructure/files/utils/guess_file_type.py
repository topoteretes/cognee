import io
from pathlib import Path
from typing import BinaryIO, Optional, Any
import filetype
from tempfile import SpooledTemporaryFile
from filetype.types.base import Type


class FileTypeException(Exception):
    """
    Represents an exception for file type errors.

    This exception is raised when an invalid file type is encountered. It includes a custom
    message to describe the error.

    Parameters:
    -----------

        - message (str): The message describing the exception error.
    """

    message: str

    def __init__(self, message: str):
        self.message = message


def guess_file_type(file: BinaryIO, name: Optional[str] = None) -> filetype.Type:
    """
    Guess the file type from the given binary file stream.

    If the file type cannot be determined from content, attempts to infer from extension.
    If still unable to determine, raise a FileTypeException with an appropriate message.

    Parameters:
    -----------

        - file (BinaryIO): A binary file stream to analyze for determining the file type.

    Returns:
    --------

        - filetype.Type: The guessed file type, represented as filetype.Type.
    """

    # Note: If file has .txt or .text extension, consider it a plain text file as filetype.guess may not detect it properly
    # as it contains no magic number encoding
    ext = None
    if isinstance(file, str):
        ext = Path(file).suffix
    elif name is not None:
        ext = Path(name).suffix

    if ext in [".txt", ".text"]:
        file_type = Type("text/plain", "txt")
        return file_type

    file_type = filetype.guess(file)

    # If file type could not be determined consider it a plain text file as they don't have magic number encoding
    if file_type is None:
        file_type = Type("text/plain", "txt")

    if file_type is None:
        raise FileTypeException(f"Unknown file detected: {file.name}.")

    return file_type
