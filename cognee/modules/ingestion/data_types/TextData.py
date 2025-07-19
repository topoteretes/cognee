from typing import BinaryIO
from contextlib import asynccontextmanager
from cognee.infrastructure.data.utils.extract_keywords import extract_keywords
from .IngestionData import IngestionData


def create_text_data(data: str):
    return TextData(data)


class TextData(IngestionData):
    data: str = None
    metadata = None

    def __init__(self, data: BinaryIO):
        self.data = data

    def get_identifier(self):
        import hashlib

        content_bytes = self.data.encode("utf-8")
        content_hash = hashlib.md5(content_bytes).hexdigest()

        return "text/plain" + "_" + content_hash

    def get_metadata(self):
        self.ensure_metadata()

        return self.metadata

    def ensure_metadata(self):
        if self.metadata is None:
            import hashlib

            keywords = extract_keywords(self.data)
            content_bytes = self.data.encode("utf-8")
            content_hash = hashlib.md5(content_bytes).hexdigest()

            self.metadata = {
                "keywords": keywords,
                "content_hash": content_hash,
                "content_type": "text/plain",
                "mime_type": "text/plain",
                "extension": "txt",
                "file_size": len(content_bytes),
            }

    @asynccontextmanager
    async def get_data(self):
        yield self.data
