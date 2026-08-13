from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Optional
from uuid import UUID


@dataclass
class LoaderResult:
    """Rich loader output for loaders that own more than text extraction.

    ``load()`` normally returns the stored derived-text path as a plain str.
    A loader that also owns the record's identity and routing (dlt: the
    manifest's stable data_id and the ``system_metadata`` route stamp) returns
    this instead; ``ingest_data`` pins the record to ``data_id`` and stamps
    ``system_metadata`` exactly as it does for pinned ``DataItem``s.
    """

    file_path: str
    data_id: Optional[UUID] = None
    system_metadata: Optional[dict] = None


class LoaderInterface(ABC):
    """
    Base interface for all file loaders in cognee.

    This interface follows cognee's established pattern for database adapters,
    ensuring consistent behavior across all loader implementations.
    """

    # Unique name identifier for this loader.
    loader_name: ClassVar[str]

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """
        List of file extensions this loader supports.

        Returns:
            List of extensions without the leading dot (e.g., ['txt', 'md'])
        """
        pass

    @property
    @abstractmethod
    def supported_mime_types(self) -> list[str]:
        """
        List of MIME types this loader supports.

        Returns:
            List of MIME type strings (e.g., ['text/plain', 'application/pdf'])
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
    async def load(self, file_path: str, **kwargs: Any) -> str:
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
