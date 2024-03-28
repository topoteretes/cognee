from typing import BinaryIO, TypedDict
import filetype
from pypdf import PdfReader
from .extract_keywords import extract_keywords

class FileTypeException(Exception):
    message: str

    def __init__(self, message: str):
        self.message = message


class FileMetadata(TypedDict):
    name: str
    mime_type: str
    extension: str
    keywords: list[str]

def get_file_metadata(file: BinaryIO) -> FileMetadata:
    file_type = filetype.guess(file)

    if file_type is None:
        raise FileTypeException("Unknown file detected.")

    keywords: list = []

    if file_type.extension == "pdf":
        reader = PdfReader(stream = file)
        pages = list(reader.pages[:3])
        text = "\n".join([page.extract_text().strip() for page in pages])
        keywords = extract_keywords(text)

    file_path = file.name
    file_name = file_path.split("/")[-1].split(".")[0]

    return FileMetadata(
        name = file_name,
        file_path = file_path,
        mime_type = file_type.mime,
        extension = file_type.extension,
        keywords = keywords
    )
