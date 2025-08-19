from abc import ABC, abstractmethod
from typing import List, Optional, Any


class LoaderInterface(ABC):
    """
    Base interface for all file loaders in cognee.

    This interface follows cognee's established pattern for database adapters,
    ensuring consistent behavior across all loader implementations.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        List of file extensions this loader supports.

        Returns:
            List of extensions including the dot (e.g., ['.txt', '.md'])
        """
        pass

    @property
    @abstractmethod
    def supported_mime_types(self) -> List[str]:
        """
        List of MIME types this loader supports.

        Returns:
            List of MIME type strings (e.g., ['text/plain', 'application/pdf'])
        """
        pass

    @property
    @abstractmethod
    def loader_name(self) -> str:
        """
        Unique name identifier for this loader.

        Returns:
            String identifier used for registration and configuration
        """
        pass

    @abstractmethod
    def can_handle(self, extension: str, mime_type: str) -> bool:
        """
        Check if this loader can handle the given file.

        Args:
            extension: File extension
            mime_type: MIME type of the file

        Returns:
            True if this loader can process the file, False otherwise
        """
        pass

    @abstractmethod
    async def load(self, file_path: str, **kwargs):
        """
        Load and process the file, returning standardized result.

        Args:
            file_path: Path to the file to be processed
            file_stream: If file stream is provided it will be used to process file instead
            **kwargs: Additional loader-specific configuration

        Raises:
            Exception: If file cannot be processed
        """
        pass
