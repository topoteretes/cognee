from typing import BinaryIO, AsyncGenerator
from contextlib import asynccontextmanager
import hashlib
from .IngestionData import IngestionData


def create_text_data(data: str) -> "TextData":
    return TextData(data)


class TextData(IngestionData):
    data: str = None
    metadata: dict = None

    def __init__(self, data: BinaryIO) -> None:
        self.data = data

    def get_identifier(self) -> str:
        metadata = self.get_metadata()

        return metadata["content_hash"]

    def get_metadata(self) -> dict:
        self.ensure_metadata()

        return self.metadata

    async def aget_identifier(self) -> str:
        metadata = await self.aget_metadata()

        return metadata["content_hash"]

    async def aget_metadata(self) -> dict:
        # Text is already in memory: no I/O to move off the event loop.
        self.ensure_metadata()

        return self.metadata

    def ensure_metadata(self) -> None:
        if self.metadata is None:
            self.metadata = {}

        data_contents = self.data.encode("utf-8")
        hash_contents = hashlib.md5(data_contents).hexdigest()
        self.metadata["name"] = "text_" + hash_contents + ".txt"
        self.metadata["content_hash"] = hash_contents
        # Describe the payload the same way reading the stored ".txt" back would
        # (``guess_file_type`` resolves a .txt extension to exactly this type).
        # Ingestion builds ``Data`` rows straight from this dict, so a partial
        # one would drop the row's extension, mime type and size.
        self.metadata["mime_type"] = "text/plain"
        self.metadata["extension"] = "txt"
        self.metadata["file_size"] = len(data_contents)
        self.metadata.setdefault("file_path", None)

    @asynccontextmanager
    async def get_data(self) -> AsyncGenerator[str, None]:
        yield self.data
