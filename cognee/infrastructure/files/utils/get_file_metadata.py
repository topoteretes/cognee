from typing import BinaryIO, TypedDict
from .guess_file_type import guess_file_type


class FileMetadata(TypedDict):
    name: str
    mime_type: str
    extension: str

def get_file_metadata(file: BinaryIO) -> FileMetadata:
    """Get metadata from a file"""
    file.seek(0)
    file_type = guess_file_type(file)

    file_path = file.name
    file_name = file_path.split("/")[-1].split(".")[0] if file_path else None

    return FileMetadata(
        name = file_name,
        file_path = file_path,
        mime_type = file_type.mime,
        extension = file_type.extension,
    )
