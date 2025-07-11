import os
from typing import Optional
from contextlib import asynccontextmanager
from cognee.infrastructure.files import get_file_metadata, FileMetadata
from cognee.infrastructure.utils import run_sync
from .IngestionData import IngestionData


def create_s3_binary_data(s3_path: str, name: Optional[str] = None) -> "S3BinaryData":
    return S3BinaryData(s3_path, name=name)


class S3BinaryData(IngestionData):
    name: Optional[str] = None
    s3_path: str = None
    metadata: Optional[FileMetadata] = None

    def __init__(self, s3_path: str, name: Optional[str] = None):
        self.s3_path = s3_path
        self.name = name

    def get_identifier(self):
        metadata = self.get_metadata()
        return metadata["content_hash"]

    def get_metadata(self):
        run_sync(self.ensure_metadata())
        return self.metadata

    async def ensure_metadata(self):
        if self.metadata is None:
            from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

            file_dir_path = os.path.dirname(self.s3_path)
            file_path = os.path.basename(self.s3_path)

            file_storage = S3FileStorage(file_dir_path)

            async with file_storage.open(file_path, "rb") as file:
                self.metadata = await get_file_metadata(file)

            if self.metadata.get("name") is None:
                self.metadata["name"] = self.name or file_path

    @asynccontextmanager
    async def get_data(self):
        from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

        file_dir_path = os.path.dirname(self.s3_path)
        file_path = os.path.basename(self.s3_path)

        file_storage = S3FileStorage(file_dir_path)

        async with file_storage.open(file_path, "rb") as file:
            yield file
