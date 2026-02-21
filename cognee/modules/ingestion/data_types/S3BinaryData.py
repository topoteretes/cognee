import os
#from tracemalloc import start
from typing import Optional
from contextlib import asynccontextmanager
from cognee.infrastructure.files import get_file_metadata, FileMetadata
from cognee.infrastructure.utils import run_sync
from .IngestionData import IngestionData

import logging
import time

logger = logging.getLogger(__name__)

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
            logger.debug("Opening S3 file for metadata: %s", self.s3_path)
            from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

            file_dir_path = os.path.dirname(self.s3_path)
            file_path = os.path.basename(self.s3_path)

            file_storage = S3FileStorage(file_dir_path)

            try:
                async with file_storage.open(file_path, "rb") as file:
                    start = time.perf_counter()
                    self.metadata = await get_file_metadata(file)
                    elapsed = time.perf_counter() - start
                    if elapsed > 2:
                        logger.warning("Slow S3 metadata read: %s took %.2fs", self.s3_path, elapsed)
                    else:
                        logger.debug("S3 metadata read completed in %.2fs: %s", elapsed, self.s3_path)
            except Exception as e:
                logger.error("Failed to read S3 metadata: %s, error: %s", self.s3_path, str(e))
                raise
            
            logger.info(
                "Loaded S3 metadata: path=%s size=%s hash=%s",
                self.s3_path,
                self.metadata.get("size"),
                self.metadata.get("content_hash"),)

            if self.metadata.get("name") is None:
                self.metadata["name"] = self.name or file_path

    @asynccontextmanager
    async def get_data(self):
        from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

        file_dir_path = os.path.dirname(self.s3_path)
        file_path = os.path.basename(self.s3_path)

        file_storage = S3FileStorage(file_dir_path)

        logger.debug("Opening S3 stream: %s", self.s3_path)

        try:
            async with file_storage.open(file_path, "rb") as file:
                yield file
            logger.info("Closed S3 stream: %s", self.s3_path)

        except Exception as e:
            logger.error("Failed to open S3 stream: %s, error: %s", self.s3_path, str(e))
            raise