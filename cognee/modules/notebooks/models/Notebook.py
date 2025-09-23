import json
import nbformat
import asyncio
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
from cognee.infrastructure.files.storage.get_file_storage import get_file_storage
from cognee.base_config import get_base_config


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
    ) -> "Notebook":
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
            Notebook instance
        """
        # Generate a cache key based on the zip URL
        content_hash = generate_content_hash(zip_url, notebook_filename)

        # Download and extract the zip file to tutorial_data/{content_hash}
        try:
            extracted_cache_dir = await download_and_extract_zip(
                url=zip_url,
                cache_dir_name=f"tutorial_data/{content_hash}",
                version_or_hash=content_hash,
                force=force,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download tutorial zip from {zip_url}") from e

        # Use cache system to access the notebook file
        from cognee.shared.cache import cache_file_exists, read_cache_file

        notebook_file_path = f"{extracted_cache_dir}/{notebook_filename}"

        # Check if the notebook file exists in cache
        if not await cache_file_exists(notebook_file_path):
            raise FileNotFoundError(f"Notebook file '{notebook_filename}' not found in zip")

        # Read and parse the notebook using cache system
        async with await read_cache_file(notebook_file_path, encoding="utf-8") as f:
            notebook_content = await asyncio.to_thread(f.read)
        notebook = cls.from_ipynb_string(notebook_content, owner_id, name, deletable)

        # Update file paths in notebook cells to point to actual cached data files
        await cls._update_file_paths_in_cells(notebook, extracted_cache_dir)

        return notebook

    @staticmethod
    async def _update_file_paths_in_cells(notebook: "Notebook", cache_dir: str) -> None:
        """
        Update file paths in code cells to use actual cached data files.
        Works with both local filesystem and S3 storage.

        Args:
            notebook: Parsed Notebook instance with cells to update
            cache_dir: Path to the cached tutorial directory containing data files
        """
        import re
        from cognee.shared.cache import list_cache_files, cache_file_exists
        from cognee.shared.logging_utils import get_logger

        logger = get_logger()

        # Look for data files in the data subdirectory
        data_dir = f"{cache_dir}/data"

        try:
            # Get all data files in the cache directory using cache system
            data_files = {}
            if await cache_file_exists(data_dir):
                file_list = await list_cache_files(data_dir)
            else:
                file_list = []

            for file_path in file_list:
                # Extract just the filename
                filename = file_path.split("/")[-1]
                # Use the file path as provided by cache system
                data_files[filename] = file_path

        except Exception as e:
            # If we can't list files, skip updating paths
            logger.error(f"Error listing data files in {data_dir}: {e}")
            return

        # Pattern to match file://data/filename patterns in code cells
        file_pattern = r'"file://data/([^"]+)"'

        def replace_path(match):
            filename = match.group(1)
            if filename in data_files:
                file_path = data_files[filename]
                # For local filesystem, preserve file:// prefix
                if not file_path.startswith("s3://"):
                    return f'"file://{file_path}"'
                else:
                    # For S3, return the S3 URL as-is
                    return f'"{file_path}"'
            return match.group(0)  # Keep original if file not found

        # Update only code cells
        updated_cells = 0
        for cell in notebook.cells:
            if cell.type == "code":
                original_content = cell.content
                # Update file paths in the cell content
                cell.content = re.sub(file_pattern, replace_path, cell.content)
                if original_content != cell.content:
                    updated_cells += 1

        # Log summary of updates (useful for monitoring)
        if updated_cells > 0:
            logger.info(f"Updated file paths in {updated_cells} notebook cells")

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
