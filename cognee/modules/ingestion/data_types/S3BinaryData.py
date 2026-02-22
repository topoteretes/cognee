import os
import logging
import time
from typing import Optional
from contextlib import asynccontextmanager
from cognee.infrastructure.files import get_file_metadata, FileMetadata
from cognee.infrastructure.utils import run_sync
from .IngestionData import IngestionData

# 1. Add logger initialization
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
            from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

            file_dir_path = os.path.dirname(self.s3_path)
            file_path = os.path.basename(self.s3_path)

            # 2. Log S3 connection details (Debug)
            logger.debug("Initializing S3 storage: bucket_path=%s", file_dir_path)
            file_storage = S3FileStorage(file_dir_path)

            try:
                # 3. Log start of operation
                logger.info("Fetching S3 metadata for: %s", self.s3_path)
                
                async with file_storage.open(file_path, "rb") as file:
                    start_time = time.perf_counter()
                    self.metadata = await get_file_metadata(file)
                    elapsed = time.perf_counter() - start_time
                    
                    # 4. Log operation duration and performance (Warning/Debug)
                    if elapsed > 2.0:
                        logger.warning("Slow S3 metadata read: %s took %.2fs", self.s3_path, elapsed)
                    else:
                        logger.debug("S3 metadata read completed in %.2fs", elapsed)

                # 5. Log successful operation with size/hash (Info)
                logger.info(
                    "Loaded S3 metadata: path=%s size=%s bytes hash=%s",
                    self.s3_path,
                    self.metadata.get("size"),
                    self.metadata.get("content_hash"),
                )

            except Exception as e:
                # 6. Log errors with appropriate context (Error)
                logger.error("Failed to read S3 metadata: %s, error: %s", self.s3_path, str(e), exc_info=True)
                raise

            if self.metadata.get("name") is None:
                self.metadata["name"] = self.name or file_path

    @asynccontextmanager
    async def get_data(self):
        from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

        file_dir_path = os.path.dirname(self.s3_path)
        file_path = os.path.basename(self.s3_path)
        file_storage = S3FileStorage(file_dir_path)

        logger.debug("Opening S3 stream: %s", self.s3_path)
        start_time = time.perf_counter()

        try:
            async with file_storage.open(file_path, "rb") as file:
                yield file
            
            elapsed = time.perf_counter() - start_time
            logger.info("Successfully streamed S3 file: %s in %.2fs", self.s3_path, elapsed)

        except Exception as e:
            logger.error("S3 stream operation failed: %s, error: %s", self.s3_path, str(e), exc_info=True)
            raise