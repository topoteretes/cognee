from typing import BinaryIO
import filetype
from .is_text_content import is_text_content


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


class TxtFileType(filetype.Type):
    """
    Represents a text file type with specific MIME and extension properties.

    Public methods:
    - match: Determines whether a given buffer matches the text file type.
    """

    MIME = "text/plain"
    EXTENSION = "txt"

    def __init__(self):
        super(TxtFileType, self).__init__(mime=TxtFileType.MIME, extension=TxtFileType.EXTENSION)

    def match(self, buf):
        """
        Determine if the given buffer contains text content.

        Parameters:
        -----------

            - buf: The buffer to check for text content.

        Returns:
        --------

            Returns True if the buffer is identified as text content, otherwise False.
        """
        return is_text_content(buf)


txt_file_type = TxtFileType()

filetype.add_type(txt_file_type)


def guess_file_type(file: BinaryIO) -> filetype.Type:
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
    file_type = filetype.guess(file)

    # If file type could not be determined consider it a plain text file as they don't have magic number encoding
    if file_type is None:
        from filetype.types.base import Type

        file_type = Type("text/plain", "txt")

    if file_type is None:
        raise FileTypeException(f"Unknown file detected: {file.name}.")

    return file_type
