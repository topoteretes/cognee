import os
from typing import Optional
from contextlib import asynccontextmanager
from cognee.infrastructure.files import get_file_metadata, FileMetadata
from cognee.infrastructure.utils.run_sync import run_sync
from cognee.infrastructure.files.storage.utils import get_scheme_with_separator
from .IngestionData import IngestionData


# Create a cloud(s3, gcs, etc.) binary data object
def create_cloud_binary_data(cloud_file_path: str, name: Optional[str] = None) -> "CloudBinaryData":
    return CloudBinaryData(cloud_file_path, name=name)


class CloudBinaryData(IngestionData):
    name: Optional[str] = None
    cloud_file_path: str = None
    metadata: Optional[FileMetadata] = None

    def __init__(self, cloud_file_path: str, name: Optional[str] = None):
        self.cloud_file_path = cloud_file_path
        self.name = name

    def get_identifier(self):
        metadata = self.get_metadata()
        return metadata["content_hash"]

    def get_metadata(self):
        run_sync(self.ensure_metadata())
        return self.metadata

    async def ensure_metadata(self):
        if self.metadata is None:
            from cognee.infrastructure.files.storage import StorageProviderRegistry

            file_dir_path = os.path.dirname(self.cloud_file_path)
            file_path = os.path.basename(self.cloud_file_path)

            scheme_with_separator = get_scheme_with_separator(self.cloud_file_path)
            cloud_storage_cls = StorageProviderRegistry.get_provider_by_cloud_scheme(
                scheme_with_separator
            )
            cloud_storage = cloud_storage_cls(file_dir_path)

            async with cloud_storage.open(file_path, "rb") as file:
                self.metadata = await get_file_metadata(file)

            if self.metadata.get("name") is None:
                self.metadata["name"] = self.name or file_path

    @asynccontextmanager
    async def get_data(self):
        from cognee.infrastructure.files.storage import StorageProviderRegistry

        file_dir_path = os.path.dirname(self.cloud_file_path)
        file_path = os.path.basename(self.cloud_file_path)

        scheme_with_separator = get_scheme_with_separator(self.cloud_file_path)
        cloud_storage_cls = StorageProviderRegistry.get_provider_by_cloud_scheme(
            scheme_with_separator
        )
        cloud_storage = cloud_storage_cls(file_dir_path)

        async with cloud_storage.open(file_path, "rb") as file:
            yield file
