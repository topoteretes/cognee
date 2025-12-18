import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4, uuid5, NAMESPACE_OID
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
import tempfile
import shutil
import importlib

from cognee.modules.notebooks.methods.create_tutorial_notebooks import (
    create_tutorial_notebooks,
    _parse_cell_index,
    _get_cell_type,
    _extract_markdown_heading,
    _get_cell_name,
    _format_tutorial_name,
    _load_tutorial_cells,
)

from cognee.modules.notebooks.models.Notebook import Notebook, NotebookCell
from cognee.shared.logging_utils import get_logger

create_tutorial_notebooks_module = importlib.import_module(
    "cognee.modules.notebooks.methods.create_tutorial_notebooks"
)

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
def temp_tutorials_dir():
    """Create a temporary tutorials directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    tutorials_dir = temp_dir / "tutorials"
    tutorials_dir.mkdir(parents=True)
    yield tutorials_dir
    shutil.rmtree(temp_dir)


class TestTutorialNotebookHelperFunctions:
    """Test cases for helper functions used in tutorial notebook creation."""

    def test_parse_cell_index_valid(self):
        """Test parsing valid cell index from filename."""
        assert _parse_cell_index("cell-0.md") == 0
        assert _parse_cell_index("cell-1.py") == 1
        assert _parse_cell_index("cell-123.md") == 123
        assert _parse_cell_index("cell-999.py") == 999

    def test_parse_cell_index_invalid(self):
        """Test parsing invalid cell index returns -1."""
        assert _parse_cell_index("not-a-cell.md") == -1
        assert _parse_cell_index("cell.md") == -1
        assert _parse_cell_index("cell-.md") == -1
        assert _parse_cell_index("") == -1

    def test_get_cell_type_markdown(self):
        """Test cell type detection for markdown files."""
        assert _get_cell_type(Path("cell-1.md")) == "markdown"
        assert _get_cell_type(Path("test.MD")) == "markdown"

    def test_get_cell_type_code(self):
        """Test cell type detection for Python files."""
        assert _get_cell_type(Path("cell-1.py")) == "code"
        assert _get_cell_type(Path("test.PY")) == "code"

    def test_get_cell_type_unsupported(self):
        """Test error handling for unsupported file types."""
        with pytest.raises(ValueError, match="Unsupported cell file type"):
            _get_cell_type(Path("cell-1.txt"))

    def test_extract_markdown_heading_single_hash(self):
        """Test extracting heading from markdown with single #."""
        content = "# My Heading\nSome content here"
        assert _extract_markdown_heading(content) == "My Heading"

    def test_extract_markdown_heading_multiple_hash(self):
        """Test extracting heading from markdown with multiple #."""
        content = "## Subheading\nSome content"
        assert _extract_markdown_heading(content) == "Subheading"

    def test_extract_markdown_heading_with_whitespace(self):
        """Test extracting heading with leading/trailing whitespace."""
        content = "  #   Heading with spaces  \nContent"
        assert _extract_markdown_heading(content) == "Heading with spaces"

    def test_extract_markdown_heading_no_heading(self):
        """Test extracting heading when no heading exists."""
        content = "Just some regular text\nNo heading here"
        assert _extract_markdown_heading(content) is None

    def test_extract_markdown_heading_empty(self):
        """Test extracting heading from empty content."""
        assert _extract_markdown_heading("") is None

    def test_get_cell_name_code_cell(self):
        """Test cell name generation for code cells."""
        cell_file = Path("cell-1.py")
        content = "print('Hello, World!')"
        assert _get_cell_name(cell_file, "code", content) == "Code Cell"

    def test_get_cell_name_markdown_with_heading(self):
        """Test cell name generation for markdown cells with heading."""
        cell_file = Path("cell-1.md")
        content = "# My Tutorial Title\nSome content"
        assert _get_cell_name(cell_file, "markdown", content) == "My Tutorial Title"

    def test_get_cell_name_markdown_no_heading(self):
        """Test cell name generation for markdown cells without heading."""
        cell_file = Path("cell-1.md")
        content = "Just some text without heading"
        assert _get_cell_name(cell_file, "markdown", content) == "cell-1"

    def test_format_tutorial_name_simple(self):
        """Test formatting simple tutorial directory name."""
        assert _format_tutorial_name("cognee-basics") == "Cognee basics - tutorial 🧠"

    def test_format_tutorial_name_with_underscores(self):
        """Test formatting tutorial name with underscores."""
        assert _format_tutorial_name("python_development") == "Python development - tutorial 🧠"

    def test_format_tutorial_name_mixed(self):
        """Test formatting tutorial name with mixed separators."""
        assert _format_tutorial_name("my-tutorial_name") == "My tutorial name - tutorial 🧠"


class TestLoadTutorialCells:
    """Test cases for loading cells from tutorial directories."""

    def test_load_tutorial_cells_sorted_order(self, temp_tutorials_dir):
        """Test that cells are loaded in sorted order by index."""
        # Create cells out of order
        (temp_tutorials_dir / "cell-3.md").write_text("# Third")
        (temp_tutorials_dir / "cell-1.md").write_text("# First")
        (temp_tutorials_dir / "cell-2.py").write_text("print('second')")

        cells = _load_tutorial_cells(temp_tutorials_dir)

        assert len(cells) == 3
        assert cells[0].name == "First"
        assert cells[1].name == "Code Cell"
        assert cells[2].name == "Third"

    def test_load_tutorial_cells_skips_non_cell_files(self, temp_tutorials_dir):
        """Test that non-cell files are skipped."""
        (temp_tutorials_dir / "cell-1.md").write_text("# First")
        (temp_tutorials_dir / "config.json").write_text('{"name": "test"}')
        (temp_tutorials_dir / "README.md").write_text("# Readme")
        (temp_tutorials_dir / "data").mkdir()
        (temp_tutorials_dir / "data" / "file.txt").write_text("data")

        cells = _load_tutorial_cells(temp_tutorials_dir)

        assert len(cells) == 1
        assert cells[0].name == "First"

    def test_load_tutorial_cells_skips_unsupported_extensions(self, temp_tutorials_dir):
        """Test that unsupported file extensions are skipped."""
        (temp_tutorials_dir / "cell-1.md").write_text("# First")
        (temp_tutorials_dir / "cell-2.txt").write_text("Text file")
        (temp_tutorials_dir / "cell-3.py").write_text("print('code')")

        cells = _load_tutorial_cells(temp_tutorials_dir)

        assert len(cells) == 2
        assert cells[0].name == "First"
        assert cells[1].name == "Code Cell"

    def test_load_tutorial_cells_empty_directory(self, temp_tutorials_dir):
        """Test loading cells from empty directory."""
        cells = _load_tutorial_cells(temp_tutorials_dir)
        assert len(cells) == 0

    def test_load_tutorial_cells_preserves_content(self, temp_tutorials_dir):
        """Test that cell content is preserved correctly."""
        markdown_content = "# My Heading\n\nSome content here."
        code_content = "import cognee\nprint('Hello')"

        (temp_tutorials_dir / "cell-1.md").write_text(markdown_content)
        (temp_tutorials_dir / "cell-2.py").write_text(code_content)

        cells = _load_tutorial_cells(temp_tutorials_dir)

        assert len(cells) == 2
        assert cells[0].content == markdown_content
        assert cells[0].type == "markdown"
        assert cells[1].content == code_content
        assert cells[1].type == "code"


class TestCreateTutorialNotebooks:
    """Test cases for the main create_tutorial_notebooks function."""

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_success_with_config(
        self, mock_session, temp_tutorials_dir
    ):
        """Test successful creation of tutorial notebooks with config.json."""
        import json

        user_id = uuid4()

        # Create a tutorial directory with cells and config.json
        tutorial_dir = temp_tutorials_dir / "test-tutorial"
        tutorial_dir.mkdir()
        (tutorial_dir / "cell-1.md").write_text("# Introduction\nWelcome to the tutorial")
        (tutorial_dir / "cell-2.py").write_text("print('Hello')")
        (tutorial_dir / "config.json").write_text(
            json.dumps({"name": "Custom Tutorial Name", "deletable": False})
        )

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            await create_tutorial_notebooks(user_id, mock_session)

        # Verify notebook was added to session
        assert mock_session.add.call_count == 1
        added_notebook = mock_session.add.call_args[0][0]

        assert isinstance(added_notebook, Notebook)
        assert added_notebook.owner_id == user_id
        assert added_notebook.name == "Custom Tutorial Name"
        assert len(added_notebook.cells) == 2
        assert added_notebook.deletable is False

        # Verify deterministic ID generation based on config name
        expected_id = uuid5(NAMESPACE_OID, name="Custom Tutorial Name")
        assert added_notebook.id == expected_id

        # Verify commit was called
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_success_without_config(
        self, mock_session, temp_tutorials_dir
    ):
        """Test successful creation of tutorial notebooks without config.json (fallback)."""
        user_id = uuid4()

        # Create a tutorial directory with cells but no config.json
        tutorial_dir = temp_tutorials_dir / "test-tutorial"
        tutorial_dir.mkdir()
        (tutorial_dir / "cell-1.md").write_text("# Introduction\nWelcome to the tutorial")
        (tutorial_dir / "cell-2.py").write_text("print('Hello')")

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            await create_tutorial_notebooks(user_id, mock_session)

        # Verify notebook was added to session
        assert mock_session.add.call_count == 1
        added_notebook = mock_session.add.call_args[0][0]

        assert isinstance(added_notebook, Notebook)
        assert added_notebook.owner_id == user_id
        assert added_notebook.name == "Test tutorial - tutorial 🧠"
        assert len(added_notebook.cells) == 2
        assert added_notebook.deletable is False  # Default for tutorials

        # Verify deterministic ID generation
        expected_id = uuid5(NAMESPACE_OID, name="Test tutorial - tutorial 🧠")
        assert added_notebook.id == expected_id

        # Verify commit was called
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_multiple_tutorials(
        self, mock_session, temp_tutorials_dir
    ):
        """Test creation of multiple tutorial notebooks."""
        user_id = uuid4()

        # Create two tutorial directories
        tutorial1 = temp_tutorials_dir / "tutorial-one"
        tutorial1.mkdir()
        (tutorial1 / "cell-1.md").write_text("# Tutorial One")

        tutorial2 = temp_tutorials_dir / "tutorial-two"
        tutorial2.mkdir()
        (tutorial2 / "cell-1.md").write_text("# Tutorial Two")

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            await create_tutorial_notebooks(user_id, mock_session)

        # Verify both notebooks were added
        assert mock_session.add.call_count == 2
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_skips_empty_tutorials(
        self, mock_session, temp_tutorials_dir
    ):
        """Test that tutorials with no cells are skipped."""
        user_id = uuid4()

        # Create tutorial directory without cells
        tutorial_dir = temp_tutorials_dir / "empty-tutorial"
        tutorial_dir.mkdir()

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            await create_tutorial_notebooks(user_id, mock_session)

        # Verify no notebooks were added
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_skips_hidden_directories(
        self, mock_session, temp_tutorials_dir
    ):
        """Test that hidden directories (starting with .) are skipped."""
        user_id = uuid4()

        # Create hidden tutorial directory
        hidden_tutorial = temp_tutorials_dir / ".hidden-tutorial"
        hidden_tutorial.mkdir()
        (hidden_tutorial / "cell-1.md").write_text("# Hidden")

        # Create visible tutorial directory
        visible_tutorial = temp_tutorials_dir / "visible-tutorial"
        visible_tutorial.mkdir()
        (visible_tutorial / "cell-1.md").write_text("# Visible")

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            await create_tutorial_notebooks(user_id, mock_session)

        # Verify only visible tutorial was added
        assert mock_session.add.call_count == 1
        added_notebook = mock_session.add.call_args[0][0]
        assert added_notebook.name == "Visible tutorial - tutorial 🧠"

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_missing_directory(self, mock_session):
        """Test handling when tutorials directory doesn't exist."""
        user_id = uuid4()

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = Path("/nonexistent/tutorials/dir")

            await create_tutorial_notebooks(user_id, mock_session)

        # Verify no notebooks were added and no commit
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_empty_directory(
        self, mock_session, temp_tutorials_dir
    ):
        """Test handling when tutorials directory is empty."""
        user_id = uuid4()

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            await create_tutorial_notebooks(user_id, mock_session)

        # Verify no notebooks were added
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_handles_cell_loading_error(
        self, mock_session, temp_tutorials_dir
    ):
        """Test that errors loading individual cells don't stop notebook creation."""
        user_id = uuid4()

        tutorial_dir = temp_tutorials_dir / "test-tutorial"
        tutorial_dir.mkdir()
        (tutorial_dir / "cell-1.md").write_text("# Valid Cell")
        # Create a file that will cause an error (invalid extension that passes filter)
        invalid_file = tutorial_dir / "cell-2.invalid"
        invalid_file.write_text("Invalid content")

        with patch.object(create_tutorial_notebooks_module, "_load_tutorial_cells") as mock_load:
            # Simulate error loading one cell but others succeed
            mock_load.return_value = [
                NotebookCell(id=uuid4(), type="markdown", name="Valid Cell", content="# Valid Cell")
            ]

            with patch.object(
                create_tutorial_notebooks_module, "_get_tutorials_directory"
            ) as mock_get_dir:
                mock_get_dir.return_value = temp_tutorials_dir

                await create_tutorial_notebooks(user_id, mock_session)

        # Verify notebook was still created with valid cells
        assert mock_session.add.call_count == 1
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_handles_tutorial_error_gracefully(
        self, mock_session, temp_tutorials_dir
    ):
        """Test that errors in one tutorial don't prevent others from being created."""
        user_id = uuid4()

        # Create two tutorials - one will fail, one will succeed
        tutorial1 = temp_tutorials_dir / "working-tutorial"
        tutorial1.mkdir()
        (tutorial1 / "cell-1.md").write_text("# Working")

        tutorial2 = temp_tutorials_dir / "broken-tutorial"
        tutorial2.mkdir()
        # Create a file that will cause an error when trying to determine cell type
        (tutorial2 / "cell-1.txt").write_text("Invalid")

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            await create_tutorial_notebooks(user_id, mock_session)

        # Verify working tutorial was created
        assert mock_session.add.call_count == 1
        added_notebook = mock_session.add.call_args[0][0]
        assert added_notebook.name == "Working tutorial - tutorial 🧠"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_deterministic_ids(
        self, mock_session, temp_tutorials_dir
    ):
        """Test that tutorial notebooks have deterministic IDs based on name."""
        user_id = uuid4()

        tutorial_dir = temp_tutorials_dir / "test-tutorial"
        tutorial_dir.mkdir()
        (tutorial_dir / "cell-1.md").write_text("# Test")

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            # Create notebooks twice
            await create_tutorial_notebooks(user_id, mock_session)
            first_notebook = mock_session.add.call_args[0][0]
            first_id = first_notebook.id

            # Reset mocks
            mock_session.add.reset_mock()
            mock_session.commit.reset_mock()

            await create_tutorial_notebooks(user_id, mock_session)
            second_notebook = mock_session.add.call_args[0][0]
            second_id = second_notebook.id

            # IDs should be the same (deterministic)
            assert first_id == second_id
            assert first_id == uuid5(NAMESPACE_OID, name="Test tutorial - tutorial 🧠")

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_with_config_deletable(
        self, mock_session, temp_tutorials_dir
    ):
        """Test that deletable flag from config.json is respected."""
        import json

        user_id = uuid4()

        tutorial_dir = temp_tutorials_dir / "test-tutorial"
        tutorial_dir.mkdir()
        (tutorial_dir / "cell-1.md").write_text("# Test")
        (tutorial_dir / "config.json").write_text(
            json.dumps({"name": "Test Tutorial", "deletable": True})
        )

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            await create_tutorial_notebooks(user_id, mock_session)

        added_notebook = mock_session.add.call_args[0][0]
        assert added_notebook.deletable is True

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_config_missing_name(
        self, mock_session, temp_tutorials_dir
    ):
        """Test that missing name in config.json falls back to formatted directory name."""
        import json

        user_id = uuid4()

        tutorial_dir = temp_tutorials_dir / "test-tutorial"
        tutorial_dir.mkdir()
        (tutorial_dir / "cell-1.md").write_text("# Test")
        (tutorial_dir / "config.json").write_text(json.dumps({"deletable": False}))

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            await create_tutorial_notebooks(user_id, mock_session)

        added_notebook = mock_session.add.call_args[0][0]
        assert added_notebook.name == "Test tutorial - tutorial 🧠"

    @pytest.mark.asyncio
    async def test_create_tutorial_notebooks_invalid_config_json(
        self, mock_session, temp_tutorials_dir
    ):
        """Test that invalid config.json is handled gracefully."""
        user_id = uuid4()

        tutorial_dir = temp_tutorials_dir / "test-tutorial"
        tutorial_dir.mkdir()
        (tutorial_dir / "cell-1.md").write_text("# Test")
        (tutorial_dir / "config.json").write_text("{ invalid json }")

        with patch.object(
            create_tutorial_notebooks_module, "_get_tutorials_directory"
        ) as mock_get_dir:
            mock_get_dir.return_value = temp_tutorials_dir

            # Should not raise, should fall back to formatted name
            await create_tutorial_notebooks(user_id, mock_session)

        added_notebook = mock_session.add.call_args[0][0]
        assert added_notebook.name == "Test tutorial - tutorial 🧠"


class TestNotebookFromIpynbString:
    """Test cases for Notebook.from_ipynb_string (legacy method, still used)."""

    def test_notebook_from_ipynb_string_success(self):
        """Test successful creation of notebook from JSON string."""
        import json

        sample_notebook = {
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
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        notebook_json = json.dumps(sample_notebook)
        user_id = uuid4()

        notebook = Notebook.from_ipynb_string(
            notebook_content=notebook_json, owner_id=user_id, name="String Test Notebook"
        )

        assert notebook.owner_id == user_id
        assert notebook.name == "String Test Notebook"
        assert len(notebook.cells) == 2
        assert notebook.cells[0].type == "markdown"
        assert notebook.cells[1].type == "code"

    def test_notebook_from_ipynb_string_invalid_json(self):
        """Test error handling for invalid JSON."""
        user_id = uuid4()
        invalid_json = "{ invalid json content"

        from nbformat.reader import NotJSONError

        with pytest.raises(NotJSONError):
            Notebook.from_ipynb_string(notebook_content=invalid_json, owner_id=user_id)
