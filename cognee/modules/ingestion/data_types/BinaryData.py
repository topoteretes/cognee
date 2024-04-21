from typing import BinaryIO
from cognee.infrastructure.files import get_file_metadata, FileMetadata
from .IngestionData import IngestionData

def create_binary_data(data: BinaryIO):
    return BinaryData(data)

class BinaryData(IngestionData):
    data: BinaryIO = None
    metadata: FileMetadata = None

    def __init__(self, data: BinaryIO):
        self.data = data

    def get_identifier(self):
        metadata = self.get_metadata()

        return metadata["mime_type"] + "_" + "|".join(metadata["keywords"])

    def get_metadata(self):
        self.ensure_metadata()

        return self.metadata

    def ensure_metadata(self):
        if self.metadata is None:
            self.metadata = get_file_metadata(self.data)

    def get_data(self):
        return self.data
