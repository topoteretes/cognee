import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from botocore.exceptions import ClientError, NoCredentialsError

from cognee.infrastructure.files import FileMetadata, get_file_metadata
from cognee.infrastructure.utils.run_sync import run_sync
from cognee.shared.logging_utils import get_logger

from .IngestionData import IngestionData

logger = get_logger(__name__)

# Threshold in seconds above which S3 operations are logged as slow
S3_SLOW_OPERATION_THRESHOLD_SEC = 30.0


def create_s3_binary_data(s3_path: str, name: Optional[str] = None) -> "S3BinaryData":
    return S3BinaryData(s3_path, name=name)


class S3BinaryData(IngestionData):
    name: Optional[str] = None
    s3_path: str = None
    metadata: Optional[FileMetadata] = None

    def __init__(self, s3_path: str, name: Optional[str] = None):
        self.s3_path = s3_path
        self.name = name

    def get_identifier(self) -> str:
        metadata = self.get_metadata()
        return metadata["content_hash"]

    def get_metadata(self) -> Optional[FileMetadata]:
        run_sync(self.ensure_metadata())
        return self.metadata

    async def ensure_metadata(self) -> None:
        if self.metadata is not None:
            return

        from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

        file_dir_path = os.path.dirname(self.s3_path)
        file_path = os.path.basename(self.s3_path)

        logger.debug(
            "Fetching S3 metadata",
            extra={"s3_path": self.s3_path, "file_path": file_path},
        )

        start_time = time.perf_counter()
        try:
            file_storage = S3FileStorage(file_dir_path)
            async with file_storage.open(file_path, "rb") as file:
                self.metadata = await get_file_metadata(file)
        except (OSError, ValueError, ClientError, NoCredentialsError) as error:
            logger.error(
                "S3 metadata fetch failed",
                extra={
                    "s3_path": self.s3_path,
                    "file_path": file_path,
                    "error": str(error),
                },
                exc_info=True,
            )
            raise

        elapsed = time.perf_counter() - start_time
        duration_sec = round(elapsed, 3)
        if elapsed > S3_SLOW_OPERATION_THRESHOLD_SEC:
            logger.warning(
                "S3 metadata fetch slow",
                extra={
                    "s3_path": self.s3_path,
                    "file_path": file_path,
                    "duration_seconds": duration_sec,
                    "threshold_seconds": S3_SLOW_OPERATION_THRESHOLD_SEC,
                },
            )
        else:
            file_size = self.metadata.get("file_size") if self.metadata else None
            logger.info(
                "S3 metadata fetched",
                extra={
                    "s3_path": self.s3_path,
                    "file_path": file_path,
                    "file_size_bytes": file_size,
                    "duration_seconds": duration_sec,
                },
            )

        if self.metadata is not None and self.metadata.get("name") is None:
            self.metadata["name"] = self.name or file_path

    @asynccontextmanager
    async def get_data(self):
        from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

        file_dir_path = os.path.dirname(self.s3_path)
        file_path = os.path.basename(self.s3_path)
        file_storage = S3FileStorage(file_dir_path)
        async with file_storage.open(file_path, "rb") as file:
            yield file
