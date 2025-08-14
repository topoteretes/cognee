import os
from typing import List
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata


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

        result = await LLMGateway.transcribe_image(file_path)

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(storage_file_name, result.choices[0].message.content)

        return full_file_path
