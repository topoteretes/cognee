import os
from typing import List, Union
from pathlib import Path
from ..LoaderInterface import LoaderInterface
from ..models.LoaderResult import LoaderResult, ContentType


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

    def can_handle(self, file_path: Union[str, Path], mime_type: str = None) -> bool:
        """
        Check if this loader can handle the given file.

        Args:
            file_path: Path to the file (Path type recommended for explicit file path handling)
            mime_type: Optional MIME type

        Returns:
            True if file can be handled, False otherwise
        """
        # Convert to Path for consistent handling
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path

        # Check by extension
        ext = path_obj.suffix.lower()
        if ext in self.supported_extensions:
            return True

        # Check by MIME type
        if mime_type and mime_type in self.supported_mime_types:
            return True

        # As fallback loader, can attempt to handle any text-like file
        # This is useful when other loaders fail
        try:
            # Quick check if file appears to be text
            with open(path_obj, "rb") as f:
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

    async def load(
        self, file_path: Union[str, Path], encoding: str = "utf-8", **kwargs
    ) -> LoaderResult:
        """
        Load and process the text file.

        Args:
            file_path: Path to the file to load (Path type recommended for explicit file path handling)
            encoding: Text encoding to use (default: utf-8)
            **kwargs: Additional configuration (unused)

        Returns:
            LoaderResult containing the file content and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file cannot be decoded with specified encoding
            OSError: If file cannot be read
        """
        # Convert to Path for consistent handling
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path_obj}")

        try:
            with open(path_obj, "r", encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with fallback encoding
            if encoding == "utf-8":
                return await self.load(path_obj, encoding="latin-1", **kwargs)
            else:
                raise

        # Extract basic metadata
        file_stat = path_obj.stat()
        metadata = {
            "name": path_obj.name,
            "size": file_stat.st_size,
            "extension": path_obj.suffix,
            "encoding": encoding,
            "loader": self.loader_name,
            "lines": len(content.splitlines()) if content else 0,
            "characters": len(content),
        }

        return LoaderResult(
            content=content,
            metadata=metadata,
            content_type=ContentType.TEXT,
            source_info={"file_path": str(path_obj), "encoding": encoding},
        )
