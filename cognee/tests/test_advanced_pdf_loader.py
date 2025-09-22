import sys
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
import pytest

from cognee.infrastructure.loaders.external.advanced_pdf_loader import AdvancedPdfLoader

advanced_pdf_loader_module = sys.modules.get(
    "cognee.infrastructure.loaders.external.advanced_pdf_loader"
)


class MockElement:
    def __init__(self, category, text, metadata):
        self.category = category
        self.text = text
        self.metadata = metadata

    def to_dict(self):
        return {
            "type": self.category,
            "text": self.text,
            "metadata": self.metadata,
        }


@pytest.fixture
def loader():
    return AdvancedPdfLoader()


@pytest.mark.parametrize(
    "extension, mime_type, expected",
    [
        ("pdf", "application/pdf", True),
        ("txt", "text/plain", False),
        ("pdf", "text/plain", False),
        ("doc", "application/pdf", False),
    ],
)
def test_can_handle(loader, extension, mime_type, expected):
    """Test can_handle method can correctly identify PDF files"""
    assert loader.can_handle(extension, mime_type) == expected


@pytest.mark.asyncio
@patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.open", new_callable=mock_open)
@patch(
    "cognee.infrastructure.loaders.external.advanced_pdf_loader.get_file_metadata",
    new_callable=AsyncMock,
)
@patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.get_storage_config")
@patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.get_file_storage")
@patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.PyPdfLoader")
@patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.partition_pdf")
async def test_load_success_with_unstructured(
    mock_partition_pdf,
    mock_pypdf_loader,
    mock_get_file_storage,
    mock_get_storage_config,
    mock_get_file_metadata,
    mock_open,
    loader,
):
    """Test the main flow of using unstructured to successfully process PDF"""
    # Prepare Mock data and objects
    mock_elements = [
        MockElement(
            category="Title", text="Attention Is All You Need", metadata={"page_number": 1}
        ),
        MockElement(
            category="NarrativeText",
            text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
            metadata={"page_number": 1},
        ),
        MockElement(
            category="Table",
            text="This is a table.",
            metadata={"page_number": 2, "text_as_html": "<table><tr><td>Data</td></tr></table>"},
        ),
    ]
    mock_pypdf_loader.return_value.load = AsyncMock(return_value="/fake/path/fallback.txt")
    mock_partition_pdf.return_value = mock_elements
    mock_get_file_metadata.return_value = {"content_hash": "abc123def456"}

    mock_storage_instance = MagicMock()
    mock_storage_instance.store = AsyncMock(return_value="/stored/text_abc123def456.txt")
    mock_get_file_storage.return_value = mock_storage_instance

    mock_get_storage_config.return_value = {"data_root_directory": "/fake/data/root"}
    test_file_path = "/fake/path/document.pdf"

    # Run

    result_path = await loader.load(test_file_path)

    # Assert
    assert result_path == "/stored/text_abc123def456.txt"

    # Verify partition_pdf is called with the correct parameters
    mock_partition_pdf.assert_called_once()
    call_args, call_kwargs = mock_partition_pdf.call_args
    assert call_kwargs.get("filename") == test_file_path
    assert call_kwargs.get("strategy") == "auto"  # Default strategy

    # Verify the stored content is correct
    expected_content = "Page 1:\nAttention Is All You Need\n\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks.\n\nPage 2:\n<table><tr><td>Data</td></tr></table>\n"
    mock_storage_instance.store.assert_awaited_once_with("text_abc123def456.txt", expected_content)

    # Verify fallback is not called
    mock_pypdf_loader.assert_not_called()


@pytest.mark.asyncio
@patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.open", new_callable=mock_open)
@patch(
    "cognee.infrastructure.loaders.external.advanced_pdf_loader.get_file_metadata",
    new_callable=AsyncMock,
)
@patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.PyPdfLoader")
@patch(
    "cognee.infrastructure.loaders.external.advanced_pdf_loader.partition_pdf",
    side_effect=Exception("Unstructured failed!"),
)
async def test_load_fallback_on_unstructured_exception(
    mock_partition_pdf, mock_pypdf_loader, mock_get_file_metadata, mock_open, loader
):
    """Test fallback to PyPdfLoader when unstructured throws an exception"""
    # Prepare Mock
    mock_fallback_instance = MagicMock()
    mock_fallback_instance.load = AsyncMock(return_value="/fake/path/fallback.txt")
    mock_pypdf_loader.return_value = mock_fallback_instance
    mock_get_file_metadata.return_value = {"content_hash": "anyhash"}
    test_file_path = "/fake/path/document.pdf"

    # Run
    result_path = await loader.load(test_file_path)

    # Assert
    assert result_path == "/fake/path/fallback.txt"
    mock_partition_pdf.assert_called_once()  # Verify partition_pdf is called
    mock_fallback_instance.load.assert_awaited_once_with(test_file_path)
