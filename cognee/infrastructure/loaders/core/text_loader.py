import os
from typing import List
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata


class TextLoader(LoaderInterface):
    """
    Core text file loader that handles basic text file formats.

    This loader is always available and serves as the fallback for
    text-based files when no specialized loader is available.
    """

    @property
    def supported_extensions(self) -> List[str]:
        """Supported text file extensions."""
        return ["txt", "md", "csv", "json", "xml", "yaml", "yml", "log"]

    @property
    def supported_mime_types(self) -> List[str]:
        """Supported MIME types for text content."""
        return [
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/json",
            "text/xml",
            "application/xml",
            "text/yaml",
            "application/yaml",
        ]

    @property
    def loader_name(self) -> str:
        """Unique identifier for this loader."""
        return "text_loader"

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

    async def load(self, file_path: str, encoding: str = "utf-8", **kwargs):
        """
        Load and process the text file.

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

        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(storage_file_name, content)

        return full_file_path
