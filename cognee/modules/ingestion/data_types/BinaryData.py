from typing import BinaryIO
from contextlib import asynccontextmanager
from cognee.infrastructure.files import get_file_metadata, FileMetadata
from cognee.infrastructure.utils.run_sync import run_sync
from .IngestionData import IngestionData


def create_binary_data(data: BinaryIO):
    return BinaryData(data)


class BinaryData(IngestionData):
    name: str = None
    data: BinaryIO = None
    metadata: FileMetadata = None

    def __init__(self, data: BinaryIO, name: str = None):
        self.name = name
        self.data = data

    def get_identifier(self):
        metadata = self.get_metadata()

        return metadata["content_hash"]

    def get_metadata(self):
        run_sync(self.ensure_metadata())

        return self.metadata

    async def ensure_metadata(self):
        if self.metadata is None:
            self.metadata = await get_file_metadata(self.data, name=self.name)

            if self.metadata["name"] is None:
                self.metadata["name"] = self.name

    @asynccontextmanager
    async def get_data(self):
        yield self.data
