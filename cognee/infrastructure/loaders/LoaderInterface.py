from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Optional
from uuid import UUID


@dataclass
class LoaderResult:
    """Rich loader output for loaders that own more than text extraction.

    ``load()`` may return the stored derived-text path as a plain str, but
        returning this instead lets a loader hand back what it already knows: the
        metadata of the text it just wrote (``file_metadata``), and — for a loader
        that owns the record's identity and routing, as dlt does — the manifest's
        stable ``data_id`` and the ``system_metadata`` route stamp, which
        ``ingest_data`` applies exactly as it does for pinned ``DataItem``s.
    """

    file_path: str
    data_id: Optional[UUID] = None
    system_metadata: Optional[dict] = None
    # Metadata for the stored derived text, computed from the content while the
    # loader still had it. Lets ingestion build the Data row without re-reading
    # the file it just wrote. None means "read it back to find out".
    file_metadata: Optional[dict] = None


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
    async def load(self, file_path: str, **kwargs: Any) -> "str | LoaderResult":
        """
        Load and process the file, returning standardized result.

        Args:
            file_path: Path to the file to be processed
            file_stream: If file stream is provided it will be used to process file instead
            **kwargs: Additional loader-specific configuration

        Returns:
            The stored derived-text path, or a ``LoaderResult`` for loaders
            that also own the record's identity and route stamp (dlt).

        Raises:
            Exception: If file cannot be processed
        """
        pass
