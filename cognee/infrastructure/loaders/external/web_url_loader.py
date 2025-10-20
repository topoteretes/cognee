from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from typing import List

from cognee.modules.ingestion.exceptions.exceptions import IngestionError
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class WebUrlLoader(LoaderInterface):
    @property
    def supported_extensions(self) -> List[str]:
        """
        List of file extensions this loader supports.

        Returns:
            List of extensions including the dot (e.g., ['.txt', '.md'])
        """
        return []  # N/A, we can safely return empty since it's used in register and get_loader_info, doesn't reflect on functionality

    @property
    def supported_mime_types(self) -> List[str]:
        """
        List of MIME types this loader supports.

        Returns:
            List of MIME type strings (e.g., ['text/plain', 'application/pdf'])
        """
        return []  # N/A, we can safely return empty since it's used in register and get_loader_info, doesn't reflect on functionality

    @property
    def loader_name(self) -> str:
        """
        Unique name identifier for this loader.

        Returns:
            String identifier used for registration and configuration
        """
        return "web_url_loader"

    def can_handle(self, extension: str, mime_type: str, data_item_path: str = None) -> bool:
        """
        Check if this loader can handle the given file.

        Args:
            extension: File extension
            mime_type: MIME type of the file

        Returns:
            True if this loader can process the file, False otherwise
        """
        if data_item_path is None:
            raise IngestionError(
                "data_item_path should not be None"
            )  # TODO: Temporarily set this to default to None so that I don't update other loaders unnecessarily yet, see TODO in LoaderEngine.py
        return data_item_path.startswith(("http://", "https://"))

    async def load(self, file_path: str, **kwargs):
        """
        Load and process the file, returning standardized result.

        Args:
            file_path: Path to the file to be processed (already saved by fetcher)
            file_stream: If file stream is provided it will be used to process file instead
            **kwargs: Additional loader-specific configuration

        Returns:
            file path to the stored file
        Raises:
            Exception: If file cannot be processed
        """

        return file_path
