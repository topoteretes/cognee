from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path
from .models.LoaderResult import LoaderResult


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
    def can_handle(self, file_path: Union[str, Path], mime_type: str = None) -> bool:
        """
        Check if this loader can handle the given file.

        Args:
            file_path: Path to the file to be processed (Path type recommended for explicit file path handling)
            mime_type: Optional MIME type of the file

        Returns:
            True if this loader can process the file, False otherwise
        """
        pass

    @abstractmethod
    async def load(self, file_path: Union[str, Path], **kwargs) -> LoaderResult:
        """
        Load and process the file, returning standardized result.

        Args:
            file_path: Path to the file to be processed (Path type recommended for explicit file path handling)
            **kwargs: Additional loader-specific configuration

        Returns:
            LoaderResult containing processed content and metadata

        Raises:
            Exception: If file cannot be processed
        """
        pass

    def get_dependencies(self) -> List[str]:
        """
        Optional: Return list of required dependencies for this loader.

        Returns:
            List of package names with optional version specifications
        """
        return []

    def validate_dependencies(self) -> bool:
        """
        Check if all required dependencies are available.

        Returns:
            True if all dependencies are installed, False otherwise
        """
        for dep in self.get_dependencies():
            # Extract package name from version specification
            package_name = dep.split(">=")[0].split("==")[0].split("<")[0]
            try:
                __import__(package_name)
            except ImportError:
                return False
        return True
