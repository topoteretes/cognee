import pytest
import tempfile
import os
from pathlib import Path

from cognee.infrastructure.loaders.core.text_loader import TextLoader
from cognee.infrastructure.loaders.models.LoaderResult import ContentType


class TestTextLoader:
    """Test the TextLoader implementation."""

    @pytest.fixture
    def text_loader(self):
        """Create a TextLoader instance for testing."""
        return TextLoader()

    @pytest.fixture
    def temp_text_file(self):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file.\nIt has multiple lines.\n")
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_binary_file(self):
        """Create a temporary binary file for testing."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\x03\x04\x05")
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_loader_properties(self, text_loader):
        """Test basic loader properties."""
        assert text_loader.loader_name == "text_loader"
        assert ".txt" in text_loader.supported_extensions
        assert ".md" in text_loader.supported_extensions
        assert "text/plain" in text_loader.supported_mime_types
        assert "application/json" in text_loader.supported_mime_types

    def test_can_handle_by_extension(self, text_loader):
        """Test file handling by extension."""
        assert text_loader.can_handle("test.txt")
        assert text_loader.can_handle("test.md")
        assert text_loader.can_handle("test.json")
        assert text_loader.can_handle("test.TXT")  # Case insensitive
        assert not text_loader.can_handle("test.pdf")

    def test_can_handle_by_mime_type(self, text_loader):
        """Test file handling by MIME type."""
        assert text_loader.can_handle("test.unknown", mime_type="text/plain")
        assert text_loader.can_handle("test.unknown", mime_type="application/json")
        assert not text_loader.can_handle("test.unknown", mime_type="application/pdf")

    def test_can_handle_text_file_heuristic(self, text_loader, temp_text_file):
        """Test handling of text files by content heuristic."""
        # Remove extension to force heuristic check
        no_ext_path = temp_text_file.replace(".txt", "")
        os.rename(temp_text_file, no_ext_path)

        try:
            assert text_loader.can_handle(no_ext_path)
        finally:
            if os.path.exists(no_ext_path):
                os.unlink(no_ext_path)

    def test_cannot_handle_binary_file(self, text_loader, temp_binary_file):
        """Test that binary files are not handled."""
        assert not text_loader.can_handle(temp_binary_file)

    @pytest.mark.asyncio
    async def test_load_text_file(self, text_loader, temp_text_file):
        """Test loading a text file."""
        result = await text_loader.load(temp_text_file)

        assert isinstance(result.content, str)
        assert "This is a test file." in result.content
        assert result.content_type == ContentType.TEXT
        assert result.metadata["loader"] == "text_loader"
        assert result.metadata["name"] == os.path.basename(temp_text_file)
        assert result.metadata["lines"] == 2
        assert result.metadata["encoding"] == "utf-8"
        assert result.source_info["file_path"] == temp_text_file

    @pytest.mark.asyncio
    async def test_load_with_custom_encoding(self, text_loader):
        """Test loading with custom encoding."""
        # Create a file with latin-1 encoding
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="latin-1"
        ) as f:
            f.write("Test with åéîøü characters")
            temp_path = f.name

        try:
            result = await text_loader.load(temp_path, encoding="latin-1")
            assert "åéîøü" in result.content
            assert result.metadata["encoding"] == "latin-1"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_load_with_fallback_encoding(self, text_loader):
        """Test automatic fallback to latin-1 encoding."""
        # Create a file with latin-1 content but try to read as utf-8
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            # Write latin-1 encoded bytes that are invalid in utf-8
            f.write(b"Test with \xe5\xe9\xee\xf8\xfc characters")
            temp_path = f.name

        try:
            # Should automatically fallback to latin-1
            result = await text_loader.load(temp_path)
            assert result.metadata["encoding"] == "latin-1"
            assert len(result.content) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, text_loader):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            await text_loader.load("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_load_empty_file(self, text_loader):
        """Test loading an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Create empty file
            temp_path = f.name

        try:
            result = await text_loader.load(temp_path)
            assert result.content == ""
            assert result.metadata["lines"] == 0
            assert result.metadata["characters"] == 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_no_dependencies(self, text_loader):
        """Test that TextLoader has no external dependencies."""
        assert text_loader.get_dependencies() == []
        assert text_loader.validate_dependencies() is True
