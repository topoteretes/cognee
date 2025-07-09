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


class CustomPdfMatcher(filetype.Type):
    """
    Match PDF file types based on MIME type and extension.

    Public methods:
    - match

    Instance variables:
    - MIME: The MIME type of the PDF.
    - EXTENSION: The file extension of the PDF.
    """

    MIME = "application/pdf"
    EXTENSION = "pdf"

    def __init__(self):
        super(CustomPdfMatcher, self).__init__(
            mime=CustomPdfMatcher.MIME, extension=CustomPdfMatcher.EXTENSION
        )

    def match(self, buf):
        """
        Determine if the provided buffer is a PDF file.

        This method checks for the presence of the PDF signature in the buffer.

        Raises:
        - TypeError: If the buffer is not of bytes type.

        Parameters:
        -----------

            - buf: The buffer containing the data to be checked.

        Returns:
        --------

            Returns True if the buffer contains a PDF signature, otherwise returns False.
        """
        return b"PDF-" in buf


custom_pdf_matcher = CustomPdfMatcher()

filetype.add_type(custom_pdf_matcher)


def guess_file_type(file: BinaryIO) -> filetype.Type:
    """
    Guess the file type from the given binary file stream.

    If the file type cannot be determined, raise a FileTypeException with an appropriate
    message.

    Parameters:
    -----------

        - file (BinaryIO): A binary file stream to analyze for determining the file type.

    Returns:
    --------

        - filetype.Type: The guessed file type, represented as filetype.Type.
    """
    file_type = filetype.guess(file)

    if file_type is None:
        raise FileTypeException(f"Unknown file detected: {file.name}.")

    return file_type
