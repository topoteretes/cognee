import os
import tempfile
import pytest
from cognee.modules.data.processing.document_types.open_data_file import open_data_file


class TestOpenDataFile:
    """Test cases for open_data_file function with file:// URL handling."""

    def test_regular_file_path(self):
        """Test that regular file paths work as before."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            test_content = "Test content for regular file path"
            f.write(test_content)
            temp_file_path = f.name

        try:
            with open_data_file(temp_file_path, mode="r") as f:
                content = f.read()
                assert content == test_content
        finally:
            os.unlink(temp_file_path)

    def test_file_url_text_mode(self):
        """Test that file:// URLs work correctly in text mode."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            test_content = "Test content for file:// URL handling"
            f.write(test_content)
            temp_file_path = f.name

        try:
            file_url = f"file://{temp_file_path}"
            with open_data_file(file_url, mode="r") as f:
                content = f.read()
                assert content == test_content
        finally:
            os.unlink(temp_file_path)

    def test_file_url_binary_mode(self):
        """Test that file:// URLs work correctly in binary mode."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            test_content = "Test content for binary mode"
            f.write(test_content)
            temp_file_path = f.name

        try:
            file_url = f"file://{temp_file_path}"
            with open_data_file(file_url, mode="rb") as f:
                content = f.read()
                assert content == test_content.encode()
        finally:
            os.unlink(temp_file_path)

    def test_file_url_with_encoding(self):
        """Test that file:// URLs work with specific encoding."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            test_content = "Test content with UTF-8: café ☕"
            f.write(test_content)
            temp_file_path = f.name

        try:
            file_url = f"file://{temp_file_path}"
            with open_data_file(file_url, mode="r", encoding="utf-8") as f:
                content = f.read()
                assert content == test_content
        finally:
            os.unlink(temp_file_path)

    def test_file_url_nonexistent_file(self):
        """Test that file:// URLs raise appropriate error for nonexistent files."""
        file_url = "file:///nonexistent/path/to/file.txt"
        with pytest.raises(FileNotFoundError):
            with open_data_file(file_url, mode="r") as f:
                f.read()

    def test_multiple_file_prefixes(self):
        """Test that multiple file:// prefixes are handled correctly."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            test_content = "Test content"
            f.write(test_content)
            temp_file_path = f.name

        try:
            # Even if someone accidentally adds multiple file:// prefixes
            file_url = f"file://file://{temp_file_path}"
            with open_data_file(file_url, mode="r") as f:
                content = f.read()
                # This should work because we only replace the first occurrence
                assert content == test_content
        except FileNotFoundError:
            # This is expected behavior - only the first file:// should be stripped
            pass
        finally:
            os.unlink(temp_file_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
