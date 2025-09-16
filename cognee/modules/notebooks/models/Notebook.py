import json
import nbformat
from nbformat.notebooknode import NotebookNode
from typing import List, Literal, Optional, cast, Tuple
from uuid import uuid4, UUID as UUID_t
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timezone
from fastapi.encoders import jsonable_encoder
from sqlalchemy import Boolean, Column, DateTime, JSON, UUID, String, TypeDecorator
from sqlalchemy.orm import mapped_column, Mapped
from pathlib import Path

from cognee.infrastructure.databases.relational import Base
from cognee.shared.cache import (
    download_and_extract_zip,
    get_tutorial_data_dir,
    generate_content_hash,
)


class NotebookCell(BaseModel):
    id: UUID_t
    type: Literal["markdown", "code"]
    name: str
    content: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class NotebookCellList(TypeDecorator):
    impl = JSON
    cache_ok = True

    def process_bind_param(self, notebook_cells, dialect):
        if notebook_cells is None:
            return []
        return [
            json.dumps(jsonable_encoder(cell)) if isinstance(cell, NotebookCell) else cell
            for cell in notebook_cells
        ]

    def process_result_value(self, cells_json_list, dialect):
        if cells_json_list is None:
            return []
        return [NotebookCell(**json.loads(json_string)) for json_string in cells_json_list]


class Notebook(Base):
    __tablename__ = "notebooks"

    id: Mapped[UUID_t] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    owner_id: Mapped[UUID_t] = mapped_column(UUID(as_uuid=True), index=True)

    name: Mapped[str] = mapped_column(String, nullable=False)

    cells: Mapped[List[NotebookCell]] = mapped_column(NotebookCellList, nullable=False)

    deletable: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    @classmethod
    async def from_ipynb_zip_url(
        cls,
        zip_url: str,
        owner_id: UUID_t,
        notebook_filename: str = "tutorial.ipynb",
        name: Optional[str] = None,
        deletable: bool = True,
        force: bool = False,
    ) -> Tuple["Notebook", Path]:
        """
        Create a Notebook instance from a remote zip file containing notebook + data files.

        Args:
            zip_url: Remote URL to fetch the .zip file from
            owner_id: UUID of the notebook owner
            notebook_filename: Name of the .ipynb file within the zip
            name: Optional custom name for the notebook
            deletable: Whether the notebook can be deleted
            force: If True, re-download even if already cached

        Returns:
            Tuple of (Notebook instance, Path to data directory)
        """
        # Generate a cache key based on the zip URL
        content_hash = generate_content_hash(zip_url, notebook_filename)
        tutorial_cache_dir = get_tutorial_data_dir() / content_hash

        # Download and extract the zip file
        try:
            download_and_extract_zip(
                url=zip_url, cache_dir=tutorial_cache_dir, version_or_hash=content_hash, force=force
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download tutorial zip from {zip_url}") from e

        # Find the notebook file in the extracted content (immediate directory only)
        notebook_path = tutorial_cache_dir / notebook_filename

        if not notebook_path or not notebook_path.exists():
            raise FileNotFoundError(f"Notebook file '{notebook_filename}' not found in zip")

        # Read and parse the notebook
        notebook_content = notebook_path.read_text(encoding="utf-8")
        notebook = cls.from_ipynb_string(notebook_content, owner_id, name, deletable)

        # Update file paths in code cells to use actual cached data files
        cls._update_file_paths_in_cells(notebook, tutorial_cache_dir)

        return notebook, tutorial_cache_dir

    @staticmethod
    def _update_file_paths_in_cells(notebook: "Notebook", cache_dir: Path) -> None:
        """
        Update file paths in code cells to use actual cached data files.

        Args:
            notebook: Parsed Notebook instance with cells to update
            cache_dir: Path to the cached tutorial directory containing data files
        """
        import re

        # Parse the notebook to find actual data files
        data_dir = cache_dir / "data"
        if not data_dir.exists():
            return

        # Get all data files in the cache directory
        data_files = {}
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                # Map filename to actual absolute path
                data_files[file_path.name] = str(file_path)

        # Pattern to match file://data/filename patterns in code cells
        file_pattern = r'"file://data/([^"]+)"'

        def replace_path(match):
            filename = match.group(1)
            if filename in data_files:
                # Return the actual absolute path to the cached file
                return f'"file://{data_files[filename]}"'
            return match.group(0)  # Keep original if file not found

        # Update only code cells
        for cell in notebook.cells:
            if cell.type == "code":
                # Update file paths in the cell content
                cell.content = re.sub(file_pattern, replace_path, cell.content)

    @classmethod
    def from_ipynb_string(
        cls,
        notebook_content: str,
        owner_id: UUID_t,
        name: Optional[str] = None,
        deletable: bool = True,
    ) -> "Notebook":
        """
        Create a Notebook instance from Jupyter notebook string content.

        Args:
            notebook_content: Raw Jupyter notebook content as string
            owner_id: UUID of the notebook owner
            name: Optional custom name for the notebook
            deletable: Whether the notebook can be deleted

        Returns:
            Notebook instance ready to be saved to database
        """
        # Parse and validate the Jupyter notebook using nbformat
        # Note: nbformat.reads() has loose typing, so we cast to NotebookNode
        jupyter_nb = cast(
            NotebookNode, nbformat.reads(notebook_content, as_version=nbformat.NO_CONVERT)
        )

        # Convert Jupyter cells to NotebookCell objects
        cells = []
        for jupyter_cell in jupyter_nb.cells:
            # Each cell is also a NotebookNode with dynamic attributes
            cell = cast(NotebookNode, jupyter_cell)
            # Skip raw cells as they're not supported in our model
            if cell.cell_type == "raw":
                continue

            # Get the source content
            content = cell.source

            # Generate a name based on content or cell index
            cell_name = cls._generate_cell_name(cell)

            # Map cell types (jupyter uses "code"/"markdown", we use same)
            cell_type = "code" if cell.cell_type == "code" else "markdown"

            cells.append(NotebookCell(id=uuid4(), type=cell_type, name=cell_name, content=content))

        # Extract notebook name from metadata if not provided
        if name is None:
            kernelspec = jupyter_nb.metadata.get("kernelspec", {})
            name = kernelspec.get("display_name") or kernelspec.get("name", "Imported Notebook")

        return cls(id=uuid4(), owner_id=owner_id, name=name, cells=cells, deletable=deletable)

    @staticmethod
    def _generate_cell_name(jupyter_cell: NotebookNode) -> str:
        """Generate a meaningful name for a notebook cell using nbformat cell."""
        if jupyter_cell.cell_type == "markdown":
            # Try to extract a title from markdown headers
            content = jupyter_cell.source

            lines = content.strip().split("\n")
            if lines and lines[0].startswith("#"):
                # Extract header text, clean it up
                header = lines[0].lstrip("#").strip()
                return header[:50] if len(header) > 50 else header
            else:
                return "Markdown Cell"
        else:
            return "Code Cell"
