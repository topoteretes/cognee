from typing import BinaryIO
from contextlib import asynccontextmanager
import hashlib
from .IngestionData import IngestionData


def create_text_data(data: str):
    return TextData(data)


class TextData(IngestionData):
    data: str = None
    metadata = None

    def __init__(self, data: BinaryIO):
        self.data = data

    def get_identifier(self):
        metadata = self.get_metadata()

        return metadata["content_hash"]

    def get_metadata(self):
        self.ensure_metadata()

        return self.metadata

    def ensure_metadata(self):
        if self.metadata is None:
            self.metadata = {}

        data_contents = self.data.encode("utf-8")
        hash_contents = hashlib.md5(data_contents).hexdigest()
        self.metadata["name"] = "text_" + hash_contents + ".txt"
        self.metadata["content_hash"] = hash_contents

    @asynccontextmanager
    async def get_data(self):
        yield self.data
