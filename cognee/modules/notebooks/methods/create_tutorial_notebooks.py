from pathlib import Path
from uuid import NAMESPACE_OID, UUID, uuid5, uuid4
from typing import List, Optional, Dict, Any
import re
import json
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.shared.logging_utils import get_logger
from cognee.root_dir import ROOT_DIR

from ..models.Notebook import Notebook, NotebookCell

logger = get_logger()


def _get_tutorials_directory() -> Path:
    """Get the path to the tutorials directory."""
    return ROOT_DIR / "modules" / "notebooks" / "tutorials"


def _parse_cell_index(filename: str) -> int:
    """Extract cell index from filename like 'cell-0.md' or 'cell-123.py'."""
    match = re.search(r"cell-(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


def _get_cell_type(file_path: Path) -> str:
    """Determine cell type from file extension."""
    extension = file_path.suffix.lower()
    if extension == ".md":
        return "markdown"
    elif extension == ".py":
        return "code"
    else:
        raise ValueError(f"Unsupported cell file type: {extension}")


def _extract_markdown_heading(content: str) -> str | None:
    """Extract the first markdown heading from content."""
    for line in content.splitlines():
        line = line.strip()
        # Match lines starting with one or more # followed by space and text
        match = re.match(r"^#+\s+(.+)$", line)
        if match:
            return match.group(1).strip()
    return None


def _get_cell_name(cell_file: Path, cell_type: str, content: str) -> str:
    """Get the appropriate name for a cell."""
    if cell_type == "code":
        return "Code Cell"
    elif cell_type == "markdown":
        heading = _extract_markdown_heading(content)
        if heading:
            return heading
    # Fallback to filename stem
    return cell_file.stem


def _load_tutorial_cells(tutorial_dir: Path) -> List[NotebookCell]:
    """Load all cells from a tutorial directory, sorted by cell index."""
    cells = []

    cell_files = [
        file_path
        for file_path in tutorial_dir.iterdir()
        if file_path.is_file()
        and file_path.name.startswith("cell-")
        and file_path.suffix in [".md", ".py"]
    ]

    cell_files.sort(key=lambda file_path: _parse_cell_index(file_path.name))

    for cell_file in cell_files:
        try:
            cell_type = _get_cell_type(cell_file)
            content = cell_file.read_text(encoding="utf-8")
            cell_name = _get_cell_name(cell_file, cell_type, content)

            cells.append(
                NotebookCell(
                    id=uuid4(),
                    type=cell_type,
                    name=cell_name,
                    content=content,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to load cell {cell_file}: {e}")
            continue

    return cells


def _read_tutorial_config(tutorial_dir: Path) -> Optional[Dict[str, Any]]:
    """Read config.json from a tutorial directory if it exists."""
    config_path = tutorial_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read config.json from {tutorial_dir}: {e}")
            return None
    return None


def _format_tutorial_name(tutorial_dir_name: str) -> str:
    """Format tutorial directory name into a readable notebook name (fallback)."""

    name = tutorial_dir_name.replace("-", " ").replace("_", " ")
    return f"{name.capitalize()} - tutorial 🧠"


async def create_tutorial_notebooks(user_id: UUID, session: AsyncSession) -> None:
    """
    Create tutorial notebooks for all tutorials found in the tutorials directory.
    Each tutorial directory will become a separate notebook.
    """
    try:
        tutorials_dir = _get_tutorials_directory()

        if not tutorials_dir.exists():
            logger.warning(f"Tutorials directory not found: {tutorials_dir}")
            return

        tutorial_dirs = [
            d for d in tutorials_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

        if not tutorial_dirs:
            logger.warning(f"No tutorial directories found in {tutorials_dir}")
            return

        notebooks_to_add = []

        for tutorial_dir in tutorial_dirs:
            try:
                cells = _load_tutorial_cells(tutorial_dir)

                if not cells:
                    logger.warning(f"No cells found in tutorial directory: {tutorial_dir}")
                    continue

                config = _read_tutorial_config(tutorial_dir)

                # Use name from config.json, or fallback to formatted directory name
                if config and "name" in config:
                    notebook_name = config["name"]
                else:
                    notebook_name = _format_tutorial_name(tutorial_dir.name)
                    logger.warning(
                        f"No config.json or 'name' field found in {tutorial_dir}, "
                        f"using fallback name: {notebook_name}"
                    )

                # Use deletable flag from config.json, or default to False for tutorials
                deletable = False
                if config and "deletable" in config:
                    deletable = bool(config["deletable"])

                notebook_id = uuid5(NAMESPACE_OID, name=notebook_name)

                notebook = Notebook(
                    id=notebook_id,
                    owner_id=user_id,
                    name=notebook_name,
                    cells=cells,
                    deletable=deletable,
                )

                notebooks_to_add.append(notebook)
                logger.info(f"Created tutorial notebook: {notebook_name} with {len(cells)} cells")

            except Exception as e:
                logger.error(f"Failed to create tutorial notebook from {tutorial_dir}: {e}")
                continue

        if not notebooks_to_add:
            return

        for notebook in notebooks_to_add:
            session.add(notebook)

        await session.commit()

    except Exception as e:
        logger.error(f"Failed to create tutorial notebooks for user {user_id}: {e}")
