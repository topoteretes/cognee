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
            # Handle case where file might be closed
            if hasattr(self.data, "closed") and self.data.closed:
                # Try to reopen the file if we have a file path
                if hasattr(self.data, "name") and self.data.name:
                    try:
                        with open(self.data.name, "rb") as reopened_file:
                            self.metadata = await get_file_metadata(reopened_file)
                    except (OSError, FileNotFoundError):
                        # If we can't reopen, create minimal metadata
                        self.metadata = {
                            "name": self.name or "unknown",
                            "file_path": getattr(self.data, "name", "unknown"),
                            "extension": "txt",
                            "mime_type": "text/plain",
                            "content_hash": f"closed_file_{id(self.data)}",
                            "file_size": 0,
                        }
                else:
                    # Create minimal metadata when file is closed and no path available
                    self.metadata = {
                        "name": self.name or "unknown",
                        "file_path": "unknown",
                        "extension": "txt",
                        "mime_type": "text/plain",
                        "content_hash": f"closed_file_{id(self.data)}",
                        "file_size": 0,
                    }
            else:
                # File is still open, proceed normally
                self.metadata = await get_file_metadata(self.data)

            if self.metadata.get("name") is None:
                self.metadata["name"] = self.name

    @asynccontextmanager
    async def get_data(self):
        yield self.data
