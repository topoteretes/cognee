import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
import tempfile
import zipfile
import httpx

from cognee.modules.notebooks.methods.create_notebook import _create_tutorial_notebook
from cognee.modules.notebooks.models.Notebook import Notebook
import cognee


# Module-level fixtures available to all test classes
@pytest.fixture
def mock_session():
    """Mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
@pytest.fixture(autouse=True)
async def test_cleanup():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


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
    async def test_create_tutorial_notebook_success(self, mock_from_zip_url, mock_session):
        """Test successful tutorial notebook creation via zip."""
        user_id = uuid4()
        mock_notebook = Notebook(
            id=uuid4(), owner_id=user_id, name="Tutorial Notebook", cells=[], deletable=False
        )
        mock_data_dir = Path("/mock/data/dir")
        mock_from_zip_url.return_value = (mock_notebook, mock_data_dir)

        await _create_tutorial_notebook(user_id, mock_session)

        # Verify the notebook was created from the correct zip URL
        mock_from_zip_url.assert_called_once_with(
            zip_url="https://github.com/topoteretes/cognee/raw/notebook_tutorial/notebooks/starter_tutorial.zip",
            owner_id=user_id,
            notebook_filename="tutorial.ipynb",
            name="Python Development with Cognee Tutorial 🧠",
            deletable=False,
            force=False,
        )

        # Verify session operations
        mock_session.add.assert_called_once_with(mock_notebook)
        mock_session.commit.assert_called_once()

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

    def test_generate_cell_name_with_markdown_header(self):
        """Test cell name generation from markdown headers."""
        from nbformat.notebooknode import NotebookNode

        # Mock a markdown cell with header
        mock_cell = NotebookNode(
            {"cell_type": "markdown", "source": "# This is a Header\n\nSome content here."}
        )

        result = Notebook._generate_cell_name(mock_cell)
        assert result == "This is a Header"

    def test_generate_cell_name_with_long_header(self):
        """Test cell name generation truncates long headers."""
        from nbformat.notebooknode import NotebookNode

        long_header = "This is a very long header that should be truncated because it exceeds the fifty character limit"
        mock_cell = NotebookNode(
            {"cell_type": "markdown", "source": f"# {long_header}\n\nContent."}
        )

        result = Notebook._generate_cell_name(mock_cell)
        assert len(result) == 50
        assert result == long_header[:50]

    def test_generate_cell_name_markdown_without_header(self):
        """Test cell name generation for markdown without header."""
        from nbformat.notebooknode import NotebookNode

        mock_cell = NotebookNode(
            {"cell_type": "markdown", "source": "Just some regular markdown text without a header."}
        )

        result = Notebook._generate_cell_name(mock_cell)
        assert result == "Markdown Cell"

    def test_generate_cell_name_code_cell(self):
        """Test cell name generation for code cells."""
        from nbformat.notebooknode import NotebookNode

        mock_cell = NotebookNode(
            {"cell_type": "code", "source": 'import pandas as pd\nprint("Hello world")'}
        )

        result = Notebook._generate_cell_name(mock_cell)
        assert result == "Code Cell"


@pytest.fixture
def sample_tutorial_zip_content(sample_jupyter_notebook):
    """Create a sample zip file content with notebook and data files."""
    return {
        "notebook": sample_jupyter_notebook,
        "data_files": {
            "data/sample.txt": "This is sample tutorial data",
            "data/config.json": '{"tutorial": "configuration"}',
            "data/example.csv": "name,value\ntest,123\nexample,456",
        },
    }


def create_test_zip(zip_content, temp_dir: Path) -> Path:
    """Helper to create a test zip file."""
    zip_path = temp_dir / "test_tutorial.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Add the notebook
        zf.writestr("tutorial.ipynb", json.dumps(zip_content["notebook"]))

        # Add data files
        for file_path, content in zip_content["data_files"].items():
            zf.writestr(file_path, content)

    return zip_path


class TestTutorialNotebookZipFunctionality:
    """Test cases for zip-based tutorial functionality."""

    @pytest.mark.asyncio
    @patch("cognee.shared.cache._is_cache_valid", return_value=False)  # Force cache miss
    @patch("cognee.shared.cache.requests.head")  # Mock HEAD request for freshness check
    @patch("cognee.shared.cache.requests.get")
    async def test_notebook_from_ipynb_zip_url_success(
        self, mock_requests_get, mock_requests_head, _mock_cache_valid, sample_tutorial_zip_content
    ):
        """Test successful creation of notebook from zip URL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = create_test_zip(sample_tutorial_zip_content, temp_path)

            # Mock the requests.head call for freshness check
            mock_head_response = MagicMock()
            mock_head_response.raise_for_status = MagicMock()
            mock_head_response.headers = {}  # No freshness headers, will skip freshness check
            mock_requests_head.return_value = mock_head_response

            # Mock the requests.get call
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.iter_content = MagicMock(return_value=[zip_path.read_bytes()])
            mock_response.headers = {}  # No content headers for storage
            mock_requests_get.return_value = mock_response

            user_id = uuid4()

            # Test the zip functionality
            import sys

            notebook_module = sys.modules["cognee.modules.notebooks.models.Notebook"]
            with patch.object(notebook_module, "get_tutorial_data_dir") as mock_get_data_dir:
                mock_cache_dir = temp_path / "cache"
                mock_cache_dir.mkdir()
                mock_get_data_dir.return_value = mock_cache_dir

                notebook, _ = await Notebook.from_ipynb_zip_url(
                    zip_url="https://example.com/tutorial.zip",
                    owner_id=user_id,
                    notebook_filename="tutorial.ipynb",
                    name="Test Zip Notebook",
                )

                # Verify notebook was created correctly
                assert notebook.owner_id == user_id
                assert notebook.name == "Test Zip Notebook"
                assert len(notebook.cells) == 4  # Should skip raw cells

    @pytest.mark.asyncio
    @patch("cognee.shared.cache._is_cache_valid", return_value=False)  # Force cache miss
    @patch("cognee.shared.cache.requests.head")  # Mock HEAD request for freshness check
    @patch("cognee.shared.cache.requests.get")
    async def test_notebook_from_ipynb_zip_url_missing_notebook(
        self, mock_requests_get, mock_requests_head, _mock_cache_valid, sample_tutorial_zip_content
    ):
        """Test error handling when notebook file is missing from zip."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create zip without the expected notebook file
            zip_path = temp_path / "test.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("wrong_name.ipynb", json.dumps(sample_tutorial_zip_content["notebook"]))

            # Mock the requests.head call for freshness check
            mock_head_response = MagicMock()
            mock_head_response.raise_for_status = MagicMock()
            mock_head_response.headers = {}  # No freshness headers, will skip freshness check
            mock_requests_head.return_value = mock_head_response

            # Mock the requests.get call
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.iter_content = MagicMock(return_value=[zip_path.read_bytes()])
            mock_response.headers = {}  # No content headers for storage
            mock_requests_get.return_value = mock_response

            user_id = uuid4()

            import sys

            notebook_module = sys.modules["cognee.modules.notebooks.models.Notebook"]
            with patch.object(notebook_module, "get_tutorial_data_dir") as mock_get_data_dir:
                mock_cache_dir = temp_path / "cache"
                mock_cache_dir.mkdir()
                mock_get_data_dir.return_value = mock_cache_dir

                with pytest.raises(
                    FileNotFoundError, match="Notebook file 'tutorial.ipynb' not found in zip"
                ):
                    await Notebook.from_ipynb_zip_url(
                        zip_url="https://example.com/tutorial.zip",
                        owner_id=user_id,
                        notebook_filename="tutorial.ipynb",
                    )

    @pytest.mark.asyncio
    @patch("cognee.shared.cache.requests.get")
    async def test_notebook_from_ipynb_zip_url_download_failure(self, mock_requests_get):
        """Test error handling when zip download fails."""
        mock_requests_get.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )

        user_id = uuid4()

        with pytest.raises(RuntimeError, match="Failed to download tutorial zip"):
            await Notebook.from_ipynb_zip_url(
                zip_url="https://example.com/nonexistent.zip", owner_id=user_id
            )

    @pytest.mark.asyncio
    @patch.object(Notebook, "from_ipynb_zip_url")
    async def test_create_tutorial_notebook_zip_success(self, mock_from_zip, mock_session):
        """Test successful tutorial notebook creation with zip."""
        user_id = uuid4()
        mock_data_dir = Path("/mock/data/dir")
        mock_notebook = Notebook(
            id=uuid4(), owner_id=user_id, name="Tutorial", cells=[], deletable=False
        )

        mock_from_zip.return_value = (mock_notebook, mock_data_dir)

        await _create_tutorial_notebook(user_id, mock_session)

        # Verify zip method was called
        mock_from_zip.assert_called_once_with(
            zip_url="https://github.com/topoteretes/cognee/raw/notebook_tutorial/notebooks/starter_tutorial.zip",
            owner_id=user_id,
            notebook_filename="tutorial.ipynb",
            name="Python Development with Cognee Tutorial 🧠",
            deletable=False,
            force=False,
        )

        # Verify session operations
        mock_session.add.assert_called_once_with(mock_notebook)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(Notebook, "from_ipynb_zip_url")
    async def test_create_tutorial_notebook_with_force_refresh(self, mock_from_zip, mock_session):
        """Test tutorial notebook creation with force refresh."""
        user_id = uuid4()

        mock_notebook = Notebook(
            id=uuid4(), owner_id=user_id, name="Tutorial", cells=[], deletable=False
        )
        mock_data_dir = Path("/mock/data/dir")
        mock_from_zip.return_value = (mock_notebook, mock_data_dir)

        # Test with force refresh enabled
        await _create_tutorial_notebook(user_id, mock_session, force_refresh=True)

        # Verify zip method was called with force=True
        mock_from_zip.assert_called_once_with(
            zip_url="https://github.com/topoteretes/cognee/raw/notebook_tutorial/notebooks/starter_tutorial.zip",
            owner_id=user_id,
            notebook_filename="tutorial.ipynb",
            name="Python Development with Cognee Tutorial 🧠",
            deletable=False,
            force=True,
        )

        mock_session.add.assert_called_once_with(mock_notebook)
        mock_session.commit.assert_called_once()

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
