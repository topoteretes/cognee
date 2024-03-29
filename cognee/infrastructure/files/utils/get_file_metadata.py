from typing import BinaryIO, TypedDict
from cognee.infrastructure.data.utils.extract_keywords import extract_keywords
from .extract_text_from_file import extract_text_from_file
from .guess_file_type import guess_file_type


class FileMetadata(TypedDict):
    name: str
    mime_type: str
    extension: str
    keywords: list[str]

def get_file_metadata(file: BinaryIO) -> FileMetadata:
    file.seek(0)
    file_type = guess_file_type(file)

    file.seek(0)
    file_text = extract_text_from_file(file, file_type)
    keywords = extract_keywords(file_text)

    file_path = file.name
    file_name = file_path.split("/")[-1].split(".")[0]

    return FileMetadata(
        name = file_name,
        file_path = file_path,
        mime_type = file_type.mime,
        extension = file_type.extension,
        keywords = keywords
    )
