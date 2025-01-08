from typing import BinaryIO
from pypdf import PdfReader
import filetype


def extract_text_from_file(file: BinaryIO, file_type: filetype.Type) -> str:
    """Extract text from a file"""
    if file_type.extension == "pdf":
        reader = PdfReader(stream=file)
        pages = list(reader.pages[:3])
        return "\n".join([page.extract_text().strip() for page in pages])

    if file_type.extension == "txt":
        return file.read().decode("utf-8")
