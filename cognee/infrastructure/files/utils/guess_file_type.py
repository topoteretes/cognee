from typing import BinaryIO
import filetype

from .is_text_content import is_text_content
from .is_csv_content import is_csv_content


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


class CsvFileType(filetype.Type):
    """
    Match CSV file types based on MIME type and extension.

    Public methods:
    - match

    Instance variables:
    - MIME: The MIME type of the CSV.
    - EXTENSION: The file extension of the CSV.
    """

    MIME = "text/csv"
    EXTENSION = "csv"

    def __init__(self):
        super().__init__(mime=self.MIME, extension=self.EXTENSION)

    def match(self, buf):
        """
        Determine if the given buffer contains csv content.

        Parameters:
        -----------

            - buf: The buffer to check for csv content.

        Returns:
        --------

            Returns True if the buffer is identified as csv content, otherwise False.
        """

        return is_csv_content(buf)


csv_file_type = CsvFileType()

filetype.add_type(csv_file_type)
