import os
import logging
import time
from typing import Optional
from contextlib import asynccontextmanager
from cognee.infrastructure.files import get_file_metadata, FileMetadata
from cognee.infrastructure.utils import run_sync
from .IngestionData import IngestionData

logger = logging.getLogger(__name__)

# Define thresholds for slow operations (in seconds)
SLOW_METADATA_FETCH_THRESHOLD = 2.0  # Example: 2 seconds
SLOW_DATA_READ_THRESHOLD = 5.0  # Example: 5 seconds


def create_s3_binary_data(s3_path: str, name: Optional[str] = None) -> "S3BinaryData":
    logger.debug(f"Creating S3BinaryData for path: {s3_path}, name: {name}")
    return S3BinaryData(s3_path, name=name)


class S3BinaryData(IngestionData):
    name: Optional[str] = None
    s3_path: str = None
    metadata: Optional[FileMetadata] = None

    def __init__(self, s3_path: str, name: Optional[str] = None):
        logger.debug(f"Initializing S3BinaryData with s3_path: {s3_path}, name: {name}")
        self.s3_path = s3_path
        self.name = name

    def get_identifier(self):
        logger.debug(f"Getting identifier for s3_path: {self.s3_path}")
        metadata = self.get_metadata()  # This will ensure metadata is fetched if not already
        identifier = metadata.get("content_hash")
        logger.debug(f"Identifier for {self.s3_path}: {identifier}")
        return identifier

    def get_metadata(self):
        logger.debug(f"Getting metadata for s3_path: {self.s3_path}")
        run_sync(self.ensure_metadata())
        logger.debug(f"Metadata retrieved: {self.metadata}")
        return self.metadata

    async def ensure_metadata(self):
        if self.metadata is None:
            logger.info(f"Metadata not cached, fetching for s3_path: {self.s3_path}")
            from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

            file_dir_path = os.path.dirname(self.s3_path)
            file_path = os.path.basename(self.s3_path)

            logger.debug(f"S3 file_dir_path: {file_dir_path}, file_path: {file_path}")

            file_storage = S3FileStorage(file_dir_path)

            try:
                start_time = time.time()
                async with file_storage.open(file_path, "rb") as file:
                    # Assuming get_file_metadata can provide file size directly or through the file object
                    # If not, we might need to adjust how file size is obtained.
                    self.metadata = await get_file_metadata(file)
                    file_size = self.metadata.get("size")  # Assuming metadata contains size
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    if file_size is not None:
                        logger.info(
                            f"Successfully fetched metadata for {self.s3_path} (size: {file_size} bytes) in {elapsed_time:.2f}s"
                        )
                    else:
                        logger.info(
                            f"Successfully fetched metadata for {self.s3_path} in {elapsed_time:.2f}s"
                        )

                    if elapsed_time > SLOW_METADATA_FETCH_THRESHOLD:
                        logger.warning(
                            f"S3 metadata fetch slow for {self.s3_path}: {elapsed_time:.2f}s"
                        )

            except Exception as e:
                logger.error(f"Error fetching metadata for {self.s3_path}: {e}", exc_info=True)
                raise

            if self.metadata.get("name") is None:
                self.metadata["name"] = self.name or file_path
                logger.debug(f"Set metadata name to: {self.metadata['name']}")
        else:
            logger.debug(f"Using cached metadata for s3_path: {self.s3_path}")

    @asynccontextmanager
    async def get_data(self):
        logger.info(f"Opening S3 file for reading: {self.s3_path}")
        from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

        file_dir_path = os.path.dirname(self.s3_path)
        file_path = os.path.basename(self.s3_path)

        logger.debug(f"S3 file_dir_path: {file_dir_path}, file_path: {file_path}")

        file_storage = S3FileStorage(file_dir_path)

        try:
            start_time = time.time()
            async with file_storage.open(file_path, "rb") as file:
                logger.debug(f"Successfully opened S3 file: {self.s3_path}")
                # Attempt to get file size if possible from the file object or cached metadata
                file_size = "unknown"
                if self.metadata and "size" in self.metadata:
                    file_size = self.metadata["size"]
                elif hasattr(file, "size"):  # Check if file object has size attribute
                    file_size = file.size

                yield file

                end_time = time.time()
                elapsed_time = end_time - start_time

                if file_size != "unknown":
                    logger.debug(
                        f"Read {file_size} bytes from S3 file: {self.s3_path} in {elapsed_time:.2f}s"
                    )
                else:
                    logger.debug(f"Read data from S3 file: {self.s3_path} in {elapsed_time:.2f}s")

                if elapsed_time > SLOW_DATA_READ_THRESHOLD:
                    logger.warning(f"S3 file read slow for {self.s3_path}: {elapsed_time:.2f}s")

                logger.debug(f"Closed S3 file: {self.s3_path}")
        except Exception as e:
            logger.error(f"Error opening/reading S3 file {self.s3_path}: {e}", exc_info=True)
            raise
