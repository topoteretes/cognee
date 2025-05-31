from typing import BinaryIO
from pypdf import PdfReader
import filetype


def extract_text_from_file(file: BinaryIO, file_type: filetype.Type) -> str:
    """
    Extract text from a file based on its type.

    Supports extraction from PDF and plain text file formats. For PDF files, it reads the
    first three pages and returns the extracted text. For plain text files, it decodes the
    file content and returns it as a string. It will raise an error if the file format is
    unsupported or if there is an issue during reading.

    Parameters:
    -----------

        - file (BinaryIO): The file stream from which to extract text.
        - file_type (filetype.Type): An object that provides the type of the file, including
          its extension.

    Returns:
    --------

        - str: The extracted text as a string.
    """
    if file_type.extension == "pdf":
        reader = PdfReader(stream=file)
        pages = list(reader.pages[:3])
        return "\n".join([page.extract_text().strip() for page in pages])

    if file_type.extension == "txt":
        return file.read().decode("utf-8")
