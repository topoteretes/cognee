import os
from typing import List
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.loaders.models.LoaderResult import LoaderResult, ContentType


class TextLoader(LoaderInterface):
    """
    Core text file loader that handles basic text file formats.

    This loader is always available and serves as the fallback for
    text-based files when no specialized loader is available.
    """

    @property
    def supported_extensions(self) -> List[str]:
        """Supported text file extensions."""
        return [".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log"]

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

    def can_handle(self, file_path: str, mime_type: str = None) -> bool:
        """
        Check if this loader can handle the given file.

        Args:
            file_path: Path to the file
            mime_type: Optional MIME type

        Returns:
            True if file can be handled, False otherwise
        """
        # Check by extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.supported_extensions:
            return True

        # Check by MIME type
        if mime_type and mime_type in self.supported_mime_types:
            return True

        # As fallback loader, can attempt to handle any text-like file
        # This is useful when other loaders fail
        try:
            # Quick check if file appears to be text
            with open(file_path, "rb") as f:
                sample = f.read(512)
                # Simple heuristic: if most bytes are printable, consider it text
                if sample:
                    try:
                        sample.decode("utf-8")
                        return True
                    except UnicodeDecodeError:
                        try:
                            sample.decode("latin-1")
                            return True
                        except UnicodeDecodeError:
                            pass
        except (OSError, IOError):
            pass

        return False

    async def load(self, file_path: str, encoding: str = "utf-8", **kwargs) -> LoaderResult:
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

        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with fallback encoding
            if encoding == "utf-8":
                return await self.load(file_path, encoding="latin-1", **kwargs)
            else:
                raise

        # Extract basic metadata
        file_stat = os.stat(file_path)
        metadata = {
            "name": os.path.basename(file_path),
            "size": file_stat.st_size,
            "extension": os.path.splitext(file_path)[1],
            "encoding": encoding,
            "loader": self.loader_name,
            "lines": len(content.splitlines()) if content else 0,
            "characters": len(content),
        }

        return LoaderResult(
            content=content,
            metadata=metadata,
            content_type=ContentType.TEXT,
            source_info={"file_path": file_path, "encoding": encoding},
        )
