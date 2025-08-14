import filetype
from typing import Dict, List, Optional, Any
from .LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


class LoaderEngine:
    """
    Main loader engine for managing file loaders.

    Follows cognee's adapter pattern similar to database engines,
    providing a centralized system for file loading operations.
    """

    def __init__(self):
        """
        Initialize the loader engine.

        Args:
            default_loader_priority: Priority order for loader selection
        """
        self._loaders: Dict[str, LoaderInterface] = {}
        self._extension_map: Dict[str, List[LoaderInterface]] = {}
        self._mime_type_map: Dict[str, List[LoaderInterface]] = {}

        self.default_loader_priority = [
            "text_loader",
            "pypdf_loader",
            "image_loader",
            "audio_loader",
            "unstructured_loader",
        ]

    def register_loader(self, loader: LoaderInterface) -> bool:
        """
        Register a loader with the engine.

        Args:
            loader: LoaderInterface implementation to register

        Returns:
            True if loader was registered successfully, False otherwise
        """

        self._loaders[loader.loader_name] = loader

        # Map extensions to loaders
        for ext in loader.supported_extensions:
            ext_lower = ext.lower()
            if ext_lower not in self._extension_map:
                self._extension_map[ext_lower] = []
            self._extension_map[ext_lower].append(loader)

        # Map mime types to loaders
        for mime_type in loader.supported_mime_types:
            if mime_type not in self._mime_type_map:
                self._mime_type_map[mime_type] = []
            self._mime_type_map[mime_type].append(loader)

        logger.info(f"Registered loader: {loader.loader_name}")
        return True

    def get_loader(
        self, file_path: str, preferred_loaders: List[str] = None
    ) -> Optional[LoaderInterface]:
        """
        Get appropriate loader for a file.

        Args:
            file_path: Path to the file to be processed
            preferred_loaders: List of preferred loader names to try first

        Returns:
            LoaderInterface that can handle the file, or None if not found
        """

        file_info = filetype.guess(file_path)

        # Try preferred loaders first
        if preferred_loaders:
            for loader_name in preferred_loaders:
                if loader_name in self._loaders:
                    loader = self._loaders[loader_name]
                    if loader.can_handle(extension=file_info.extension, mime_type=file_info.mime):
                        return loader
                else:
                    raise ValueError(f"Loader does not exist: {loader_name}")

        # Try default priority order
        for loader_name in self.default_loader_priority:
            if loader_name in self._loaders:
                loader = self._loaders[loader_name]
                if loader.can_handle(extension=file_info.extension, mime_type=file_info.mime):
                    return loader
            else:
                raise ValueError(f"Loader does not exist: {loader_name}")

        return None

    async def load_file(
        self,
        file_path: str,
        file_stream: Optional[Any],
        preferred_loaders: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Load file using appropriate loader.

        Args:
            file_path: Path to the file to be processed
            preferred_loaders: List of preferred loader names to try first
            **kwargs: Additional loader-specific configuration

        Raises:
            ValueError: If no suitable loader is found
            Exception: If file processing fails
        """
        loader = self.get_loader(file_path, preferred_loaders)
        if not loader:
            raise ValueError(f"No loader found for file: {file_path}")

        logger.debug(f"Loading {file_path} with {loader.loader_name}")
        # TODO: loading needs to be reworked to work with both file streams and file locations
        return await loader.load(file_path, **kwargs)

    def get_available_loaders(self) -> List[str]:
        """
        Get list of available loader names.

        Returns:
            List of registered loader names
        """
        return list(self._loaders.keys())

    def get_loader_info(self, loader_name: str) -> Dict[str, any]:
        """
        Get information about a specific loader.

        Args:
            loader_name: Name of the loader to inspect

        Returns:
            Dictionary containing loader information
        """
        if loader_name not in self._loaders:
            return {}

        loader = self._loaders[loader_name]
        return {
            "name": loader.loader_name,
            "extensions": loader.supported_extensions,
            "mime_types": loader.supported_mime_types,
        }
