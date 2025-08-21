import sys
from unittest.mock import MagicMock, patch

from cognee.infrastructure.files.storage.utils import get_storage_type


get_storage_type_module = sys.modules.get(
    "cognee.infrastructure.files.storage.utils.get_storage_type"
)


def test_detection_by_scheme():
    """Test if the function can correctly identify the type by URL scheme"""
    assert get_storage_type("s3://my-bucket/data.csv") == "s3"
    assert get_storage_type("file:///etc/hosts") == "local"
    assert get_storage_type("http://example.com/file/example.txt") == "local"


@patch.object(get_storage_type_module, "os")
@patch.object(get_storage_type_module, "get_base_config")
def test_detection_of_local_paths_without_scheme(mock_get_base_config, mock_os):
    """Test if the function can correctly identify local paths without scheme"""
    # Mock os.getenv
    mock_os.getenv.return_value = "local"

    # Mock get_base_config
    base_config = MagicMock()
    base_config.system_root_directory = "/User/system"
    base_config.data_root_directory = "/User/data"
    mock_get_base_config.return_value = base_config

    assert get_storage_type("/home/user/documents/report.docx") == "local"
    assert get_storage_type("C:\\Users\\user\\Desktop\\image.jpg") == "local"

    mock_os.getenv.assert_called_with("STORAGE_BACKEND")
    mock_get_base_config.assert_called_with()


@patch.object(get_storage_type_module, "os")
@patch.object(get_storage_type_module, "get_base_config")
def test_s3_detection_via_environment_config(mock_get_base_config, mock_os):
    """
    Test if the function can correctly identify S3 paths when the environment is configured for S3.
    """
    # Mock os.getenv
    mock_os.getenv.return_value = "s3"

    # Mock get_base_config
    base_config = MagicMock()
    base_config.system_root_directory = "s3://my-bucket/system"
    base_config.data_root_directory = "s3://my-bucket/data"
    mock_get_base_config.return_value = base_config

    # Assert that the function returns 's3' even for local paths
    assert get_storage_type("/some/local/path") == "s3"

    # Assert that the mock was called
    mock_os.getenv.assert_called_with("STORAGE_BACKEND")
    mock_get_base_config.assert_called_once_with()
