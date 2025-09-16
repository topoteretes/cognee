import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from cognee.modules.users.methods.create_user import _create_tutorial_notebook
from cognee.modules.notebooks.models.Notebook import Notebook, NotebookCell


class TestTutorialNotebookCreation:
    """Test cases for tutorial notebook creation functionality."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = AsyncMock(spec=AsyncSession)
        session.add = MagicMock()
        session.commit = AsyncMock()
        return session

    @pytest.fixture
    def sample_jupyter_notebook(self):
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

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_notebook_from_ipynb_url_success(
        self, mock_httpx_client, sample_jupyter_notebook
    ):
        """Test successful creation of notebook from remote URL."""
        # Setup mock HTTP response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = json.dumps(sample_jupyter_notebook)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx_client.return_value = mock_client

        # Test the method
        user_id = uuid4()
        notebook = await Notebook.from_ipynb_url(
            url="https://example.com/test.ipynb",
            owner_id=user_id,
            name="Test Notebook",
            deletable=True,
        )

        # Verify the notebook was created correctly
        assert notebook.owner_id == user_id
        assert notebook.name == "Test Notebook"
        assert notebook.deletable is True
        assert len(notebook.cells) == 4  # Should skip the raw cell

        # Check cell content
        assert notebook.cells[0].type == "markdown"
        assert notebook.cells[0].name == "Tutorial Introduction"
        assert "This is a tutorial notebook." in notebook.cells[0].content

        assert notebook.cells[1].type == "code"
        assert "import cognee" in notebook.cells[1].content

        assert notebook.cells[2].type == "markdown"
        assert notebook.cells[2].name == "Step 1: Data Ingestion"

        assert notebook.cells[3].type == "code"
        assert "Add your data here" in notebook.cells[3].content

        # Verify HTTP call was made
        mock_client.get.assert_called_once_with("https://example.com/test.ipynb")

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
        assert notebook.cells[1].name == "Code Cell 2"  # Index 1 in original, but 2 in display
        assert notebook.cells[3].name == "Code Cell 4"

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
    @patch("httpx.AsyncClient")
    async def test_notebook_from_ipynb_url_http_error(self, mock_httpx_client):
        """Test error handling when HTTP request fails."""
        # Create a mock that raises HTTPStatusError when raise_for_status is called
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx_client.return_value = mock_client

        user_id = uuid4()

        with pytest.raises(httpx.HTTPStatusError):
            await Notebook.from_ipynb_url(
                url="https://example.com/nonexistent.ipynb", owner_id=user_id
            )

    @pytest.mark.asyncio
    async def test_notebook_from_ipynb_string_invalid_json(self):
        """Test error handling for invalid JSON."""
        user_id = uuid4()
        invalid_json = "{ invalid json content"

        from nbformat.reader import NotJSONError

        with pytest.raises(NotJSONError):
            Notebook.from_ipynb_string(notebook_content=invalid_json, owner_id=user_id)

    @pytest.mark.asyncio
    @patch.object(Notebook, "from_ipynb_url")
    async def test_create_tutorial_notebook_success(self, mock_from_ipynb_url, mock_session):
        """Test successful tutorial notebook creation."""
        user_id = uuid4()
        mock_notebook = Notebook(
            id=uuid4(), owner_id=user_id, name="Tutorial Notebook", cells=[], deletable=False
        )
        mock_from_ipynb_url.return_value = mock_notebook

        await _create_tutorial_notebook(user_id, mock_session)

        # Verify the notebook was created from the correct URL
        mock_from_ipynb_url.assert_called_once_with(
            url="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/notebook_tutorial/notebooks/tutorial.ipynb",
            owner_id=user_id,
            name="Python Development with Cognee Tutorial 🧠",
            deletable=False,
        )

        # Verify session operations
        mock_session.add.assert_called_once_with(mock_notebook)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(Notebook, "from_ipynb_url")
    async def test_create_tutorial_notebook_fallback_on_error(
        self, mock_from_ipynb_url, mock_session
    ):
        """Test that errors are raised when URL fetch fails."""
        user_id = uuid4()
        mock_from_ipynb_url.side_effect = Exception("Network error")

        # Should raise the exception (not catch it)
        with pytest.raises(Exception, match="Network error"):
            await _create_tutorial_notebook(user_id, mock_session)

        # Verify error handling path was taken
        mock_from_ipynb_url.assert_called_once()
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()

    def test_generate_cell_name_with_markdown_header(self):
        """Test cell name generation from markdown headers."""
        from nbformat.notebooknode import NotebookNode

        # Mock a markdown cell with header
        mock_cell = NotebookNode(
            {"cell_type": "markdown", "source": "# This is a Header\n\nSome content here."}
        )

        result = Notebook._generate_cell_name(mock_cell, 0)
        assert result == "This is a Header"

    def test_generate_cell_name_with_long_header(self):
        """Test cell name generation truncates long headers."""
        from nbformat.notebooknode import NotebookNode

        long_header = "This is a very long header that should be truncated because it exceeds the fifty character limit"
        mock_cell = NotebookNode(
            {"cell_type": "markdown", "source": f"# {long_header}\n\nContent."}
        )

        result = Notebook._generate_cell_name(mock_cell, 0)
        assert len(result) == 50
        assert result == long_header[:50]

    def test_generate_cell_name_markdown_without_header(self):
        """Test cell name generation for markdown without header."""
        from nbformat.notebooknode import NotebookNode

        mock_cell = NotebookNode(
            {"cell_type": "markdown", "source": "Just some regular markdown text without a header."}
        )

        result = Notebook._generate_cell_name(mock_cell, 2)
        assert result == "Markdown Cell 3"  # Index + 1

    def test_generate_cell_name_code_cell(self):
        """Test cell name generation for code cells."""
        from nbformat.notebooknode import NotebookNode

        mock_cell = NotebookNode(
            {"cell_type": "code", "source": 'import pandas as pd\nprint("Hello world")'}
        )

        result = Notebook._generate_cell_name(mock_cell, 4)
        assert result == "Code Cell 5"  # Index + 1


class TestTutorialNotebookIntegration:
    """Integration tests for tutorial notebook functionality."""

    @pytest.mark.asyncio
    async def test_tutorial_notebook_url_accessibility(self):
        """Test that the actual tutorial notebook URL is accessible (integration test)."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/notebook_tutorial/notebooks/tutorial.ipynb"
                )
                response.raise_for_status()

                # Verify it's valid JSON
                notebook_data = response.json()
                assert "cells" in notebook_data
                assert "nbformat" in notebook_data
                assert isinstance(notebook_data["cells"], list)

        except httpx.RequestError:
            pytest.skip("Network request failed - skipping integration test")
        except httpx.HTTPStatusError as e:
            pytest.fail(f"HTTP error accessing tutorial notebook: {e}")

    @pytest.mark.asyncio
    async def test_real_tutorial_notebook_parsing(self):
        """Test parsing the actual tutorial notebook (integration test)."""
        try:
            user_id = uuid4()
            notebook = await Notebook.from_ipynb_url(
                url="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/notebook_tutorial/notebooks/tutorial.ipynb",
                owner_id=user_id,
                name="Integration Test",
            )

            # Basic validation
            assert notebook.owner_id == user_id
            assert notebook.name == "Integration Test"
            assert len(notebook.cells) > 0

            # Should have at least some markdown and code cells
            has_markdown = any(cell.type == "markdown" for cell in notebook.cells)
            has_code = any(cell.type == "code" for cell in notebook.cells)
            assert has_markdown, "Tutorial should have markdown cells"
            assert has_code, "Tutorial should have code cells"

        except (httpx.RequestError, httpx.HTTPStatusError):
            pytest.skip("Network request failed - skipping integration test")
