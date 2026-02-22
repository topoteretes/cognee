import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from cognee.modules.ingestion.data_types.S3BinaryData import S3BinaryData

@pytest.mark.asyncio
async def test_s3_metadata_logging_success(caplog):
    """Verify that successful metadata retrieval logs info and debug messages."""
    caplog.set_level("DEBUG")
    data = S3BinaryData("s3://bucket/file.txt", name = "TestFile")

    with patch("cognee.infrastructure.files.storage.S3FileStorage.S3FileStorage"):
        with patch("cognee.modules.ingestion.data_types.S3BinaryData.get_file_metadata", new_callable = AsyncMock) as mock_meta:
            mock_meta.return_value = {"content_hash": "hash123", "size": 100}
            
            await data.ensure_metadata()
            
            assert "Fetching S3 metadata for" in caplog.text
            assert "Loaded S3 metadata: path=s3://bucket/file.txt size=100 bytes hash=hash123" in caplog.text

@pytest.mark.asyncio
async def test_s3_slow_operation_warning(caplog):
    """Verify that a slow S3 response triggers a WARNING log."""
    caplog.set_level("WARNING")
    data = S3BinaryData("s3://bucket/slow_file.txt")

    with patch("cognee.infrastructure.files.storage.S3FileStorage.S3FileStorage"):
        with patch("cognee.modules.ingestion.data_types.S3BinaryData.get_file_metadata", new_callable = AsyncMock) as mock_meta:
            # Simulate a 2.1 second delay to trigger the > 2s threshold
            async def slow_call(*args):
                await asyncio.sleep(2.1)
                return {"content_hash": "slow_hash", "size": 50}
            mock_meta.side_effect = slow_call
            
            await data.ensure_metadata()
            
            assert "Slow S3 metadata read" in caplog.text

@pytest.mark.asyncio
async def test_s3_error_logging(caplog):
    """Verify that S3 failures log an ERROR with context."""
    caplog.set_level("ERROR")
    data = S3BinaryData("s3://bucket/error_file.txt")

    with patch("cognee.infrastructure.files.storage.S3FileStorage.S3FileStorage"):
        with patch("cognee.modules.ingestion.data_types.S3BinaryData.get_file_metadata", new_callable = AsyncMock) as mock_meta:
            mock_meta.side_effect = Exception("Access Denied")
            
            with pytest.raises(Exception):
                await data.ensure_metadata()
            
            assert "Failed to read S3 metadata" in caplog.text
            assert "Access Denied" in caplog.text