import os
import tempfile
from typing import BinaryIO, Union, Optional, Any
from io import StringIO, BytesIO

from cognee.infrastructure.loaders.models.LoaderResult import LoaderResult, ContentType
from cognee.modules.ingestion.data_types import IngestionData, TextData, BinaryData
from cognee.infrastructure.files import get_file_metadata
from cognee.shared.logging_utils import get_logger


class LoaderResultToIngestionData(IngestionData):
    """
    Adapter class that wraps LoaderResult to be compatible with IngestionData interface.

    This maintains backward compatibility with existing cognee ingestion pipeline
    while enabling the new loader system.
    """

    def __init__(self, loader_result: LoaderResult, original_file_path: str = None):
        self.loader_result = loader_result
        self.original_file_path = original_file_path
        self._cached_metadata = None
        self.logger = get_logger(__name__)

    def get_identifier(self) -> str:
        """
        Get content identifier for deduplication.

        Always generates hash from content to ensure consistency with existing system.
        """
        # Always generate hash from content for consistency
        import hashlib

        content_bytes = self.loader_result.content.encode("utf-8")
        content_hash = hashlib.md5(content_bytes).hexdigest()

        # Add content type prefix for better identification
        content_type = self.loader_result.content_type.value
        return f"{content_type}_{content_hash}"

    def get_metadata(self) -> dict:
        """
        Get file metadata in the format expected by existing pipeline.

        Converts LoaderResult metadata to the format used by IngestionData.
        """
        if self._cached_metadata is not None:
            return self._cached_metadata

        # Start with loader result metadata
        metadata = self.loader_result.metadata.copy()

        # Ensure required fields are present
        if "name" not in metadata:
            if self.original_file_path:
                metadata["name"] = os.path.basename(self.original_file_path)
            else:
                # Generate name from content hash
                content_hash = self.get_identifier().split("_")[-1][:8]
                ext = metadata.get("extension", ".txt")
                metadata["name"] = f"content_{content_hash}{ext}"

        if "content_hash" not in metadata:
            # Store content hash without prefix for compatibility with deletion system
            identifier = self.get_identifier()
            if "_" in identifier:
                # Remove content type prefix (e.g., "text_abc123" -> "abc123")
                metadata["content_hash"] = identifier.split("_", 1)[-1]
            else:
                metadata["content_hash"] = identifier

        if "file_path" not in metadata and self.original_file_path:
            metadata["file_path"] = self.original_file_path

        # Add mime type if not present
        if "mime_type" not in metadata:
            ext = metadata.get("extension", "").lower()
            mime_type_map = {
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".csv": "text/csv",
                ".json": "application/json",
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            }
            metadata["mime_type"] = mime_type_map.get(ext, "application/octet-stream")

        self._cached_metadata = metadata
        return metadata

    def get_data(self) -> Union[str, BinaryIO]:
        """
        Get data content in format expected by existing pipeline.

        Returns content as string for text data or creates a file-like object
        for binary data to maintain compatibility.
        """
        if self.loader_result.content_type == ContentType.TEXT:
            return self.loader_result.content

        # For structured or binary content, return as string for now
        # The existing pipeline expects text content for processing
        return self.loader_result.content


class LoaderToIngestionAdapter:
    """
    Adapter that bridges the new loader system with existing ingestion pipeline.

    This class provides methods to process files using the loader system
    while maintaining compatibility with the existing IngestionData interface.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    async def process_file_with_loaders(
        self,
        file_path: str,
        s3fs: Optional[Any] = None,
        preferred_loaders: Optional[list] = None,
        loader_config: Optional[dict] = None,
    ) -> IngestionData:
        """
        Process a file using the loader system and return IngestionData.

        Args:
            file_path: Path to the file to process
            s3fs: S3 filesystem (for compatibility with existing code)
            preferred_loaders: List of preferred loader names
            loader_config: Configuration for specific loaders

        Returns:
            IngestionData compatible object

        Raises:
            Exception: If no loader can handle the file
        """
        from cognee.infrastructure.loaders import get_loader_engine

        try:
            # Get the loader engine
            engine = get_loader_engine()

            # Determine MIME type if possible
            mime_type = None
            try:
                import mimetypes

                mime_type, _ = mimetypes.guess_type(file_path)
            except Exception:
                pass

            # Load file using loader system
            self.logger.info(f"Processing file with loaders: {file_path}")

            # Extract loader-specific config if provided
            kwargs = {}
            if loader_config:
                # Find the first available loader that matches our preferred loaders
                loader = engine.get_loader(file_path, mime_type, preferred_loaders)
                if loader and loader.loader_name in loader_config:
                    kwargs = loader_config[loader.loader_name]

            loader_result = await engine.load_file(
                file_path, mime_type=mime_type, preferred_loaders=preferred_loaders, **kwargs
            )

            # Convert to IngestionData compatible format
            return LoaderResultToIngestionData(loader_result, file_path)

        except Exception as e:
            self.logger.warning(f"Loader system failed for {file_path}: {e}")
            # Fallback to existing classification system
            return await self._fallback_to_existing_system(file_path, s3fs)

    async def _fallback_to_existing_system(
        self, file_path: str, s3fs: Optional[Any] = None
    ) -> IngestionData:
        """
        Fallback to existing ingestion.classify() system for backward compatibility.

        This ensures that even if the loader system fails, we can still process
        files using the original classification method.
        """
        from cognee.modules.ingestion import classify

        self.logger.info(f"Falling back to existing classification system for: {file_path}")

        # Open file and classify using existing system
        if file_path.startswith("s3://"):
            if s3fs:
                with s3fs.open(file_path, "rb") as file:
                    return classify(file)
            else:
                raise ValueError("S3 file path provided but no s3fs available")
        else:
            # Handle local files and file:// URLs
            local_path = file_path.replace("file://", "")
            with open(local_path, "rb") as file:
                return classify(file)

    def is_text_content(self, data: Union[str, Any]) -> bool:
        """
        Check if the provided data is text content (not a file path).

        Args:
            data: The data to check

        Returns:
            True if data is text content, False if it's a file path
        """
        if not isinstance(data, str):
            return False

        # Check if it's a file path
        if (
            data.startswith("/")
            or data.startswith("file://")
            or data.startswith("s3://")
            or (len(data) > 1 and data[1] == ":")
        ):  # Windows drive paths
            return False

        return True

    def create_text_ingestion_data(self, content: str) -> IngestionData:
        """
        Create IngestionData for text content.

        Args:
            content: Text content to wrap

        Returns:
            IngestionData compatible object
        """

        return TextData(content)
