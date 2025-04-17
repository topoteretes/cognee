from typing import Optional
import s3fs
from cognee.infrastructure.files import get_file_metadata, FileMetadata
from .IngestionData import IngestionData


def create_s3_binary_data(
    s3_path: str, name: Optional[str] = None, s3: Optional[s3fs.S3FileSystem] = None
) -> "S3BinaryData":
    return S3BinaryData(s3_path, name=name, s3=s3)


class S3BinaryData(IngestionData):
    name: Optional[str] = None
    s3_path: str = None
    fs: s3fs.S3FileSystem = None
    metadata: Optional[FileMetadata] = None

    def __init__(
        self, s3_path: str, name: Optional[str] = None, s3: Optional[s3fs.S3FileSystem] = None
    ):
        self.s3_path = s3_path
        self.name = name
        self.fs = s3 if s3 is not None else s3fs.S3FileSystem()

    def get_identifier(self):
        metadata = self.get_metadata()
        return metadata["content_hash"]

    def get_metadata(self):
        self.ensure_metadata()
        return self.metadata

    def ensure_metadata(self):
        if self.metadata is None:
            with self.fs.open(self.s3_path, "rb") as f:
                self.metadata = get_file_metadata(f)
            if self.metadata.get("name") is None:
                self.metadata["name"] = self.name or self.s3_path.split("/")[-1]

    def get_data(self):
        return self.fs.open(self.s3_path, "rb")
