import sys
import unittest
from unittest.mock import patch, MagicMock

from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

mock_s3fs = MagicMock()
sys.modules["s3fs"] = mock_s3fs


class TestS3FileStorage(unittest.TestCase):
    @patch("cognee.infrastructure.files.storage.S3FileStorage.get_s3_config")
    def test_s3_file_storage_with_explicit_credentials(self, mock_get_s3_config):
        mock_s3fs.S3FileSystem.reset_mock()
        mock_config = MagicMock()
        mock_config.aws_access_key_id = "test_key"
        mock_config.aws_secret_access_key = "test_secret"
        mock_config.aws_session_token = "test_token"
        mock_config.aws_endpoint_url = "test_url"
        mock_config.aws_region = "test_region"
        mock_get_s3_config.return_value = mock_config

        S3FileStorage("test_path")

        mock_s3fs.S3FileSystem.assert_called_once_with(
            key="test_key",
            secret="test_secret",
            token="test_token",
            anon=False,
            endpoint_url="test_url",
            client_kwargs={"region_name": "test_region"},
        )

    @patch("cognee.infrastructure.files.storage.S3FileStorage.get_s3_config")
    def test_s3_file_storage_without_explicit_credentials(self, mock_get_s3_config):
        mock_s3fs.S3FileSystem.reset_mock()
        mock_config = MagicMock()
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_config.aws_endpoint_url = "test_url"
        mock_config.aws_region = "test_region"
        mock_get_s3_config.return_value = mock_config

        S3FileStorage("test_path")

        mock_s3fs.S3FileSystem.assert_called_once_with(
            anon=False,
            endpoint_url="test_url",
            client_kwargs={"region_name": "test_region"},
        )
