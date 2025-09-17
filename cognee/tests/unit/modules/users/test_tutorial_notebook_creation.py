import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import hashlib
import time
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
import zipfile
from cognee.shared.cache import get_tutorial_data_dir

from cognee.modules.notebooks.methods.create_notebook import _create_tutorial_notebook
from cognee.modules.notebooks.models.Notebook import Notebook
import cognee
from cognee.shared.logging_utils import get_logger

logger = get_logger()


# Module-level fixtures available to all test classes
@pytest.fixture
def mock_session():
    """Mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def sample_jupyter_notebook():
    """Sample Jupyter notebook content for testing."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Tutorial Introduction\n", "\n", "This is a tutorial notebook."],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import cognee\n", "print('Hello, Cognee!')"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 1: Data Ingestion\n", "\n", "Let's add some data."],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Add your data here\n", "# await cognee.add('data.txt')"],
            },
            {
                "cell_type": "raw",
                "metadata": {},
                "source": ["This is a raw cell that should be skipped"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


class TestTutorialNotebookCreation:
    """Test cases for tutorial notebook creation functionality."""

    @pytest.mark.asyncio
    async def test_notebook_from_ipynb_string_success(self, sample_jupyter_notebook):
        """Test successful creation of notebook from JSON string."""
        notebook_json = json.dumps(sample_jupyter_notebook)
        user_id = uuid4()

        notebook = Notebook.from_ipynb_string(
            notebook_content=notebook_json, owner_id=user_id, name="String Test Notebook"
        )

        assert notebook.owner_id == user_id
        assert notebook.name == "String Test Notebook"
        assert len(notebook.cells) == 4  # Should skip the raw cell
        assert notebook.cells[0].type == "markdown"
        assert notebook.cells[1].type == "code"

    @pytest.mark.asyncio
    async def test_notebook_cell_name_generation(self, sample_jupyter_notebook):
        """Test that cell names are generated correctly from markdown headers."""
        user_id = uuid4()
        notebook_json = json.dumps(sample_jupyter_notebook)

        notebook = Notebook.from_ipynb_string(notebook_content=notebook_json, owner_id=user_id)

        # Check markdown header extraction
        assert notebook.cells[0].name == "Tutorial Introduction"
        assert notebook.cells[2].name == "Step 1: Data Ingestion"

        # Check code cell naming
        assert notebook.cells[1].name == "Code Cell"
        assert notebook.cells[3].name == "Code Cell"

    @pytest.mark.asyncio
    async def test_notebook_from_ipynb_string_with_default_name(self, sample_jupyter_notebook):
        """Test notebook creation uses kernelspec display_name when no name provided."""
        user_id = uuid4()
        notebook_json = json.dumps(sample_jupyter_notebook)

        notebook = Notebook.from_ipynb_string(notebook_content=notebook_json, owner_id=user_id)

        assert notebook.name == "Python 3"  # From kernelspec.display_name

    @pytest.mark.asyncio
    async def test_notebook_from_ipynb_string_fallback_name(self):
        """Test fallback naming when kernelspec is missing."""
        minimal_notebook = {
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Test"]}],
            "metadata": {},  # No kernelspec
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        user_id = uuid4()
        notebook_json = json.dumps(minimal_notebook)

        notebook = Notebook.from_ipynb_string(notebook_content=notebook_json, owner_id=user_id)

        assert notebook.name == "Imported Notebook"  # Fallback name

    @pytest.mark.asyncio
    async def test_notebook_from_ipynb_string_invalid_json(self):
        """Test error handling for invalid JSON."""
        user_id = uuid4()
        invalid_json = "{ invalid json content"

        from nbformat.reader import NotJSONError

        with pytest.raises(NotJSONError):
            Notebook.from_ipynb_string(notebook_content=invalid_json, owner_id=user_id)

    @pytest.mark.asyncio
    @patch.object(Notebook, "from_ipynb_zip_url")
    async def test_create_tutorial_notebook_error_propagated(self, mock_from_zip_url, mock_session):
        """Test that errors are propagated when zip fetch fails."""
        user_id = uuid4()
        mock_from_zip_url.side_effect = Exception("Network error")

        # Should raise the exception (not catch it)
        with pytest.raises(Exception, match="Network error"):
            await _create_tutorial_notebook(user_id, mock_session)

        # Verify error handling path was taken
        mock_from_zip_url.assert_called_once()
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()

    def test_generate_cell_name_code_cell(self):
        """Test cell name generation for code cells."""
        from nbformat.notebooknode import NotebookNode

        mock_cell = NotebookNode(
            {"cell_type": "code", "source": 'import pandas as pd\nprint("Hello world")'}
        )

        result = Notebook._generate_cell_name(mock_cell)
        assert result == "Code Cell"


class TestTutorialNotebookZipFunctionality:
    """Test cases for zip-based tutorial functionality."""

    @pytest.mark.asyncio
    async def test_notebook_from_ipynb_zip_url_missing_notebook(
        self,
    ):
        """Test error handling when notebook file is missing from zip."""
        user_id = uuid4()

        with pytest.raises(
            FileNotFoundError,
            match="Notebook file 'super_random_tutorial_name.ipynb' not found in zip",
        ):
            await Notebook.from_ipynb_zip_url(
                zip_url="https://github.com/topoteretes/cognee/raw/notebook_tutorial/notebooks/starter_tutorial.zip",
                owner_id=user_id,
                notebook_filename="super_random_tutorial_name.ipynb",
            )

    @pytest.mark.asyncio
    async def test_notebook_from_ipynb_zip_url_download_failure(self):
        """Test error handling when zip download fails."""
        user_id = uuid4()
        with pytest.raises(RuntimeError, match="Failed to download tutorial zip"):
            await Notebook.from_ipynb_zip_url(
                zip_url="https://github.com/topoteretes/cognee/raw/notebook_tutorial/notebooks/nonexistent_tutorial_name.zip",
                owner_id=user_id,
            )

    @pytest.mark.asyncio
    async def test_create_tutorial_notebook_zip_success(self, mock_session):
        """Test successful tutorial notebook creation with zip."""
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        user_id = uuid4()

        # Check that tutorial data directory is empty using storage-aware method
        tutorial_data_dir_path = await get_tutorial_data_dir()
        tutorial_data_dir = Path(tutorial_data_dir_path)
        if tutorial_data_dir.exists():
            assert not any(tutorial_data_dir.iterdir()), "Tutorial data directory should be empty"

        await _create_tutorial_notebook(user_id, mock_session)

        items = list(tutorial_data_dir.iterdir())
        assert len(items) == 1, "Tutorial data directory should contain exactly one item"
        assert items[0].is_dir(), "Tutorial data directory item should be a directory"

        # Verify the structure inside the tutorial directory
        tutorial_dir = items[0]

        # Check for tutorial.ipynb file
        notebook_file = tutorial_dir / "tutorial.ipynb"
        assert notebook_file.exists(), f"tutorial.ipynb should exist in {tutorial_dir}"
        assert notebook_file.is_file(), "tutorial.ipynb should be a file"

        # Check for data subfolder with contents
        data_folder = tutorial_dir / "data"
        assert data_folder.exists(), f"data subfolder should exist in {tutorial_dir}"
        assert data_folder.is_dir(), "data should be a directory"

        data_items = list(data_folder.iterdir())
        assert len(data_items) > 0, (
            f"data folder should contain files, but found {len(data_items)} items"
        )

    @pytest.mark.asyncio
    async def test_create_tutorial_notebook_with_force_refresh(self, mock_session):
        """Test tutorial notebook creation with force refresh."""
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        user_id = uuid4()

        # Check that tutorial data directory is empty using storage-aware method
        tutorial_data_dir_path = await get_tutorial_data_dir()
        tutorial_data_dir = Path(tutorial_data_dir_path)
        if tutorial_data_dir.exists():
            assert not any(tutorial_data_dir.iterdir()), "Tutorial data directory should be empty"

        # First creation (without force refresh)
        await _create_tutorial_notebook(user_id, mock_session, force_refresh=False)

        items_first = list(tutorial_data_dir.iterdir())
        assert len(items_first) == 1, (
            "Tutorial data directory should contain exactly one item after first creation"
        )
        first_dir = items_first[0]
        assert first_dir.is_dir(), "Tutorial data directory item should be a directory"

        # Verify the structure inside the tutorial directory (first creation)
        notebook_file = first_dir / "tutorial.ipynb"
        assert notebook_file.exists(), f"tutorial.ipynb should exist in {first_dir}"
        assert notebook_file.is_file(), "tutorial.ipynb should be a file"

        data_folder = first_dir / "data"
        assert data_folder.exists(), f"data subfolder should exist in {first_dir}"
        assert data_folder.is_dir(), "data should be a directory"

        data_items = list(data_folder.iterdir())
        assert len(data_items) > 0, (
            f"data folder should contain files, but found {len(data_items)} items"
        )

        # Capture metadata from first creation

        first_creation_metadata = {}

        for file_path in first_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(first_dir)
                stat = file_path.stat()

                # Store multiple metadata points
                with open(file_path, "rb") as f:
                    content = f.read()

                first_creation_metadata[str(relative_path)] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "hash": hashlib.md5(content).hexdigest(),
                    "first_bytes": content[:100]
                    if content
                    else b"",  # First 100 bytes as fingerprint
                }

        # Wait a moment to ensure different timestamps
        time.sleep(0.1)

        # Force refresh - should create new files with different metadata
        await _create_tutorial_notebook(user_id, mock_session, force_refresh=True)

        items_second = list(tutorial_data_dir.iterdir())
        assert len(items_second) == 1, (
            "Tutorial data directory should contain exactly one item after force refresh"
        )
        second_dir = items_second[0]

        # Verify the structure is maintained after force refresh
        notebook_file_second = second_dir / "tutorial.ipynb"
        assert notebook_file_second.exists(), (
            f"tutorial.ipynb should exist in {second_dir} after force refresh"
        )
        assert notebook_file_second.is_file(), "tutorial.ipynb should be a file after force refresh"

        data_folder_second = second_dir / "data"
        assert data_folder_second.exists(), (
            f"data subfolder should exist in {second_dir} after force refresh"
        )
        assert data_folder_second.is_dir(), "data should be a directory after force refresh"

        data_items_second = list(data_folder_second.iterdir())
        assert len(data_items_second) > 0, (
            f"data folder should still contain files after force refresh, but found {len(data_items_second)} items"
        )

        # Compare metadata to ensure files are actually different
        files_with_changed_metadata = 0

        for file_path in second_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(second_dir)
                relative_path_str = str(relative_path)

                # File should exist from first creation
                assert relative_path_str in first_creation_metadata, (
                    f"File {relative_path_str} missing from first creation"
                )

                old_metadata = first_creation_metadata[relative_path_str]

                # Get new metadata
                stat = file_path.stat()
                with open(file_path, "rb") as f:
                    new_content = f.read()

                new_metadata = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "hash": hashlib.md5(new_content).hexdigest(),
                    "first_bytes": new_content[:100] if new_content else b"",
                }

                # Check if any metadata changed (indicating file was refreshed)
                metadata_changed = (
                    new_metadata["mtime"] > old_metadata["mtime"]  # Newer modification time
                    or new_metadata["hash"] != old_metadata["hash"]  # Different content hash
                    or new_metadata["size"] != old_metadata["size"]  # Different file size
                    or new_metadata["first_bytes"]
                    != old_metadata["first_bytes"]  # Different content
                )

                if metadata_changed:
                    files_with_changed_metadata += 1

        # Assert that force refresh actually updated files
        assert files_with_changed_metadata > 0, (
            f"Force refresh should have updated at least some files, but all {len(first_creation_metadata)} "
            f"files appear to have identical metadata. This suggests force refresh didn't work."
        )

        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_tutorial_zip_url_accessibility(self):
        """Test that the actual tutorial zip URL is accessible (integration test)."""
        try:
            import requests

            response = requests.get(
                "https://github.com/topoteretes/cognee/raw/notebook_tutorial/notebooks/starter_tutorial.zip",
                timeout=10,
            )
            response.raise_for_status()

            # Verify it's a valid zip file by checking headers
            assert response.headers.get("content-type") in [
                "application/zip",
                "application/octet-stream",
                "application/x-zip-compressed",
            ] or response.content.startswith(b"PK")  # Zip file signature

        except Exception:
            pytest.skip("Network request failed or zip not available - skipping integration test")
