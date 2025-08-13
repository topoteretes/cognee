import os
from typing import List
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.llm.LLMGateway import LLMGateway


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

        # async with open_data_file(file_path) as file:

        result = await LLMGateway.transcribe_image(file_path)
        return result.choices[0].message.content
