import pytest
import tempfile
import os
from pathlib import Path

from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage
from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories


class TestPathSupport:
    """Test Path type support in ingestion functions."""

    @pytest.fixture
    def temp_text_file(self):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file for Path support.\n")
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory with test files."""
        import tempfile

        temp_dir = tempfile.mkdtemp()

        # Create some test files
        for i in range(3):
            with open(os.path.join(temp_dir, f"test_{i}.txt"), "w") as f:
                f.write(f"Test file {i} content.\n")

        yield temp_dir

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_save_data_item_path_object(self, temp_text_file):
        """Test save_data_item_to_storage with Path object."""
        path_obj = Path(temp_text_file)
        result = await save_data_item_to_storage(path_obj)

        # Should return a file:// URL
        assert result.startswith("file://")
        assert str(path_obj.resolve()) in result

    @pytest.mark.asyncio
    async def test_save_data_item_string_vs_path(self, temp_text_file):
        """Test that Path object vs string path are handled consistently."""
        path_obj = Path(temp_text_file)
        string_path = str(path_obj.resolve())

        # Both should work and produce similar results
        result_path = await save_data_item_to_storage(path_obj)
        result_string = await save_data_item_to_storage(string_path)

        # Both should be file:// URLs pointing to the same file
        assert result_path.startswith("file://")
        assert result_string.startswith("file://")

        # Extract the actual file paths from the URLs
        path_from_path_obj = result_path.replace("file://", "")
        path_from_string = result_string.replace("file://", "")

        # They should resolve to the same absolute path
        assert os.path.normpath(path_from_path_obj) == os.path.normpath(path_from_string)

    @pytest.mark.asyncio
    async def test_save_data_item_text_content(self):
        """Test that plain text strings are handled as content, not paths."""
        text_content = "This is plain text content, not a file path."
        result = await save_data_item_to_storage(text_content)

        # Should create a file and return file:// URL since this is text content
        assert result.startswith("file://")

    @pytest.mark.asyncio
    async def test_resolve_data_directories_path_object(self, temp_directory):
        """Test resolve_data_directories with Path object."""
        path_obj = Path(temp_directory)
        result = await resolve_data_directories([path_obj])

        # Should return a list of Path objects for the files in the directory
        assert len(result) == 3  # We created 3 test files
        assert all(isinstance(item, Path) for item in result)
        assert all(item.suffix == ".txt" for item in result)

    @pytest.mark.asyncio
    async def test_resolve_data_directories_mixed_types(self, temp_directory, temp_text_file):
        """Test resolve_data_directories with mixed Path and string types."""
        path_obj = Path(temp_text_file)
        string_path = str(temp_text_file)
        directory_path = Path(temp_directory)

        # Mix of types
        mixed_data = [path_obj, string_path, directory_path]
        result = await resolve_data_directories(mixed_data)

        # Should have:
        # - 1 Path object (original file as Path)
        # - 1 string (original file as string)
        # - 3 Path objects (from directory expansion)
        assert len(result) == 5

        # Count types
        path_objects = [item for item in result if isinstance(item, Path)]
        string_objects = [item for item in result if isinstance(item, str)]

        assert len(path_objects) == 4  # 1 original + 3 from directory
        assert len(string_objects) == 1  # 1 original string

    @pytest.mark.asyncio
    async def test_resolve_data_directories_path_single_file(self, temp_text_file):
        """Test resolve_data_directories with a single Path file."""
        path_obj = Path(temp_text_file)
        result = await resolve_data_directories([path_obj])

        # Should return the same Path object
        assert len(result) == 1
        assert isinstance(result[0], Path)
        assert str(result[0]) == str(path_obj)
