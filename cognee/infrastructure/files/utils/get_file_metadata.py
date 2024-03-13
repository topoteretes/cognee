from typing import BinaryIO, TypedDict
import filetype
from unstructured.cleaners.core import clean
from unstructured.partition.pdf import partition_pdf
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
        elements = partition_pdf(file = file, strategy = "fast")
        keywords = extract_keywords(
            "\n".join(map(lambda element: clean(element.text), elements))
        )

    file_path = file.name
    file_name = file_path.split("/")[-1].split(".")[0]

    return FileMetadata(
        name = file_name,
        file_path = file_path,
        mime_type = file_type.mime,
        extension = file_type.extension,
        keywords = keywords
    )
