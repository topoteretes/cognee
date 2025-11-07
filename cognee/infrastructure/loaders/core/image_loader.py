import os
from typing import List

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.mineru import get_mineru_http_client
from cognee.infrastructure.mineru.http_client import MineruHTTPClientError
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


class ImageLoader(LoaderInterface):
    """
    Core image file loader that handles basic image file formats.
    """

    @property
    def supported_extensions(self) -> List[str]:
        """Supported text file extensions."""
        return [
            "png",
            "dwg",
            "xcf",
            "jpg",
            ".jpe",
            ".jpeg",
            "jpx",
            "apng",
            "gif",
            "webp",
            "cr2",
            "tif",
            "tiff",
            "bmp",
            "jxr",
            "psd",
            "ico",
            "heic",
            "avif",
        ]

    @property
    def supported_mime_types(self) -> List[str]:
        """Supported MIME types for text content."""
        return [
            "image/png",
            "image/vnd.dwg",
            "image/x-xcf",
            "image/jpeg",
            "image/jpx",
            "image/apng",
            "image/gif",
            "image/webp",
            "image/x-canon-cr2",
            "image/tiff",
            "image/bmp",
            "image/jxr",
            "image/vnd.adobe.photoshop",
            "image/vnd.microsoft.icon",
            "image/heic",
            "image/avif",
        ]

    @property
    def loader_name(self) -> str:
        """Unique identifier for this loader."""
        return "image_loader"

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

    async def load(self, file_path: str, **kwargs):
        """
        Load and process the image file.

        Args:
            file_path: Path to the file to load
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

        content = await self._transcribe_image(file_path)

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(storage_file_name, content)

        return full_file_path

    async def _transcribe_image(self, file_path: str) -> str:
        """
        Try MinerU first (if configured) before falling back to the default LLM transcription.
        """

        mineru_client = get_mineru_http_client()
        if mineru_client is not None:
            try:
                async with open_data_file(file_path, mode="rb") as image_file:
                    image_bytes = image_file.read()
                response = await mineru_client.extract_text(
                    image_bytes,
                    source_name=os.path.basename(file_path),
                )
                if response:
                    return response
            except MineruHTTPClientError as exc:
                logger.warning(
                    "MinerU transcription failed, falling back to default image transcription.",
                    extra={"error": str(exc), "file_path": file_path},
                )
            except Exception as exc:  # noqa: BLE001 - log unexpected exceptions and fallback
                logger.warning(
                    "Unexpected error while using MinerU transcription, falling back.",
                    extra={"error": str(exc), "file_path": file_path},
                )

        # Fallback path using the configured LLM provider.
        result = await LLMGateway.transcribe_image(file_path)
        message = result.choices[0].message.content if result.choices else ""
        return message or ""
