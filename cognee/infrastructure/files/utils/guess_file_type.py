import io
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO, Optional

import filetype
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


def guess_file_type(file: BinaryIO, name: str | None = None) -> filetype.Type:
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
    else:
        file_name = getattr(file, "name", None) or getattr(file, "full_name", None)
        if isinstance(file_name, str):
            ext = Path(file_name).suffix

    if ext in [".txt", ".text"]:
        file_type = Type("text/plain", "txt")
        return file_type

    if ext in [".csv"]:
        return Type("text/csv", "csv")

    if ext in [".md", ".markdown"]:
        return Type("text/markdown", "md")

    if ext in [".json"]:
        return Type("application/json", "json")

    if ext in [".xml"]:
        return Type("application/xml", "xml")

    if ext in [".yaml", ".yml"]:
        return Type("application/yaml", "yaml")

    file_type = filetype.guess(file)

    # If file type could not be determined consider it a plain text file as they don't have magic number encoding
    if file_type is None:
        file_type = Type("text/plain", "txt")

    return file_type
