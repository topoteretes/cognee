import csv
import os
from typing import Any

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface


class CsvLoader(LoaderInterface):
    """
    Core CSV file loader that handles basic CSV file formats.
    """

    loader_name = "csv_loader"

    @property
    def supported_extensions(self) -> list[str]:
        """Supported text file extensions."""
        return [
            "csv",
        ]

    @property
    def supported_mime_types(self) -> list[str]:
        """Supported MIME types for text content."""
        return [
            "text/csv",
        ]

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """
        Check if this loader can handle the given file.

        Args:
            extension: File extension
            mime_type: Optional MIME type

        Returns:
            True if file can be handled, False otherwise
        """
        if extension in self.supported_extensions and mime_type in self.supported_mime_types:
            return True

        return False

    async def load(self, file_path: str, encoding: str = "utf-8", **kwargs: Any) -> str:
        """
        Load and process the csv file.

        Args:
            file_path: Path to the file to load
            encoding: Text encoding to use (default: utf-8)
            **kwargs: Additional configuration (unused)

        Returns:
            LoaderResult containing the file content and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file cannot be decoded with specified encoding
            OSError: If file cannot be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_metadata = await get_file_metadata(f)
        # Name ingested file of current loader based on original file content hash
        storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"

        row_texts = []
        row_index = 1

        with open(file_path, encoding=encoding, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                pairs = [f"{str(k)}: {str(v)}" for k, v in row.items()]
                row_text = ", ".join(pairs)
                row_texts.append(f"Row {row_index}:\n{row_text}\n")
                row_index += 1

        content = "\n".join(row_texts)

        if not kwargs.get("persist", True):
            return content

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(storage_file_name, content)

        return full_file_path
