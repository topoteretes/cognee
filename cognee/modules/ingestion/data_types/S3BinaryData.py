import os
import time
from typing import Optional, Any
from contextlib import asynccontextmanager
from cognee.infrastructure.files import get_file_metadata, FileMetadata
from cognee.infrastructure.utils import run_sync
from .IngestionData import IngestionData
from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

# Define thresholds for slow operations (in seconds)
SLOW_METADATA_FETCH_THRESHOLD = 2.0  # Example: 2 seconds
SLOW_DATA_READ_THRESHOLD = 5.0  # Example: 5 seconds


def create_s3_binary_data(s3_path: str, name: Optional[str] = None) -> "S3BinaryData":
    """Factory function to create an S3BinaryData instance.

    Args:
        s3_path: The S3 path to the binary data.
        name: Optional name for the data.

    Returns:
        An instance of S3BinaryData.
    """
    logger.debug(f"Creating S3BinaryData for path: {s3_path}, name: {name}")
    return S3BinaryData(s3_path, name=name)


class S3BinaryData(IngestionData):
    name: str | None = None
    s3_path: str = None
    metadata: Optional[FileMetadata] = None

    def __init__(self, s3_path: str, name: Optional[str] = None):
        logger.debug(f"Initializing S3BinaryData with s3_path: {s3_path}, name: {name}")
        self.s3_path = s3_path
        self.name = name

    def get_identifier(self):
        logger.debug(f"Getting identifier for s3_path: {self.s3_path}")
        # Ensure metadata is fetched before getting identifier
        self.get_metadata()
        # Access self.metadata directly after ensuring it's fetched
        identifier = self.metadata.get("content_hash") if self.metadata else None
        logger.debug(f"Identifier for {self.s3_path}: {identifier}")
        return identifier

    def get_metadata(self):
        logger.debug(f"Getting metadata for s3_path: {self.s3_path}")
        # Use run_sync to ensure async ensure_metadata is called in sync context
        run_sync(self.ensure_metadata())
        logger.debug(f"Metadata retrieved: {self.metadata}")
        return self.metadata

    async def ensure_metadata(self):
        if self.metadata is None:
            logger.info(f"Metadata not cached, fetching for s3_path: {self.s3_path}")
            file_dir_path = os.path.dirname(self.s3_path)
            file_path = os.path.basename(self.s3_path)

            logger.debug(f"S3 file_dir_path: {file_dir_path}, file_path: {file_path}")

            file_storage = S3FileStorage(file_dir_path)

            try:
                start_time = time.time()
                async with file_storage.open(file_path, "rb") as file:
                    self.metadata = await get_file_metadata(file)
                    # Use the correct key 'file_size'
                    file_size = self.metadata.get("file_size")
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

            # Set name from metadata if not provided and if it exists in metadata
            if self.metadata and self.metadata.get("name") is None:
                self.metadata["name"] = self.name or file_path
                logger.debug(f"Set metadata name to: {self.metadata['name']}")
        else:
            logger.debug(f"Using cached metadata for s3_path: {self.s3_path}")

    @asynccontextmanager
    async def get_data(self):
        logger.info(f"Opening S3 file for reading: {self.s3_path}")

        file_dir_path = os.path.dirname(self.s3_path)
        file_path = os.path.basename(self.s3_path)

        logger.debug(f"S3 file_dir_path: {file_dir_path}, file_path: {file_path}")

        file_storage = S3FileStorage(file_dir_path)

        try:
            # Start timing before opening the file for reading
            start_time = time.time()
            async with file_storage.open(file_path, "rb") as file:
                logger.debug(f"Successfully opened S3 file: {self.s3_path}")

                # Attempt to get file size if possible from the file object or cached metadata
                file_size_val: Any = "unknown"
                if self.metadata and "file_size" in self.metadata:
                    file_size_val = self.metadata["file_size"]
                # Check if the file object has a 'size' attribute (common for file-like objects)
                elif hasattr(file, "seek") and hasattr(
                    file, "tell"
                ):  # Seekable objects might have size information
                    current_pos = file.tell()
                    file.seek(0, os.SEEK_END)  # Go to the end of the file
                    file_size_val = file.tell()  # Get the size
                    file.seek(current_pos)  # Go back to original position

                # Yield the file object to the caller
                yield file

                # Record time after the file has been used and closed (implicitly by async with)
                end_time = time.time()
                elapsed_time = end_time - start_time

                if file_size_val != "unknown":
                    logger.debug(
                        f"Read {file_size_val} bytes from S3 file: {self.s3_path} in {elapsed_time:.2f}s"
                    )
                else:
                    logger.debug(f"Read data from S3 file: {self.s3_path} in {elapsed_time:.2f}s")

                if elapsed_time > SLOW_DATA_READ_THRESHOLD:
                    logger.warning(f"S3 file read slow for {self.s3_path}: {elapsed_time:.2f}s")

                logger.debug(f"Closed S3 file: {self.s3_path}")
        except Exception as e:
            logger.error(f"Error opening/reading S3 file {self.s3_path}: {e}", exc_info=True)
            raise
