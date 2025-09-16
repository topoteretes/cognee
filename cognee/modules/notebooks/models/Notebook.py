import json
import httpx
import nbformat
from nbformat.notebooknode import NotebookNode
from typing import List, Literal, Dict, Any, Optional, cast, Tuple
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
    store_tutorial_files,
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
    async def from_ipynb_url(
        cls, url: str, owner_id: UUID_t, name: Optional[str] = None, deletable: bool = True
    ) -> "Notebook":
        """
        Create a Notebook instance from a remote Jupyter notebook (.ipynb) URL.

        Args:
            url: Remote URL to fetch the .ipynb file from
            owner_id: UUID of the notebook owner
            name: Optional custom name for the notebook (defaults to extracting from metadata)
            deletable: Whether the notebook can be deleted (defaults to True)

        Returns:
            Notebook instance ready to be saved to database
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            notebook_content = response.text

        return cls.from_ipynb_string(notebook_content, owner_id, name, deletable)

    @classmethod
    async def from_ipynb_zip_url(
        cls,
        zip_url: str,
        owner_id: UUID_t,
        notebook_filename: str = "tutorial.ipynb",
        name: Optional[str] = None,
        deletable: bool = True,
        force: bool = False,
    ) -> Tuple["Notebook", Optional[Path]]:
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
            Tuple of (Notebook instance, Path to data directory or None)
        """
        # Generate a cache key based on the zip URL
        content_hash = generate_content_hash(zip_url, notebook_filename)
        tutorial_cache_dir = get_tutorial_data_dir() / content_hash

        # Download and extract the zip file
        success, extracted_dir = download_and_extract_zip(
            url=zip_url, cache_dir=tutorial_cache_dir, version_or_hash=content_hash, force=force
        )

        if not success or not extracted_dir:
            raise RuntimeError(f"Failed to download tutorial zip from {zip_url}")

        # Find the notebook file in the extracted content
        notebook_path = None

        # Search for the notebook file recursively
        for potential_path in extracted_dir.rglob(notebook_filename):
            if potential_path.is_file():
                notebook_path = potential_path
                break

        if not notebook_path or not notebook_path.exists():
            raise FileNotFoundError(f"Notebook file '{notebook_filename}' not found in zip")

        # Read and parse the notebook
        notebook_content = notebook_path.read_text(encoding="utf-8")
        notebook = cls.from_ipynb_string(notebook_content, owner_id, name, deletable)

        # Store tutorial data files in a organized way
        data_dir = get_tutorial_data_dir() / f"{content_hash}_data"
        if store_tutorial_files(extracted_dir, data_dir):
            return notebook, data_dir
        else:
            return notebook, None

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
        for i, jupyter_cell in enumerate(jupyter_nb.cells):
            # Each cell is also a NotebookNode with dynamic attributes
            cell = cast(NotebookNode, jupyter_cell)
            # Skip raw cells as they're not supported in our model
            if cell.cell_type == "raw":
                continue

            # Get the source content
            content = cell.source

            # Generate a name based on content or cell index
            cell_name = cls._generate_cell_name(cell, i)

            # Map cell types (jupyter uses "code"/"markdown", we use same)
            cell_type = "code" if cell.cell_type == "code" else "markdown"

            cells.append(NotebookCell(id=uuid4(), type=cell_type, name=cell_name, content=content))

        # Extract notebook name from metadata if not provided
        if name is None:
            kernelspec = jupyter_nb.metadata.get("kernelspec", {})
            name = kernelspec.get("display_name") or kernelspec.get("name", "Imported Notebook")

        return cls(id=uuid4(), owner_id=owner_id, name=name, cells=cells, deletable=deletable)

    @staticmethod
    def _generate_cell_name(jupyter_cell: NotebookNode, index: int) -> str:
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
                return f"Markdown Cell {index + 1}"
        else:
            return f"Code Cell {index + 1}"
