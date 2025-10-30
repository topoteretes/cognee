import filetype
from typing import Dict, List, Optional, Any
from .LoaderInterface import LoaderInterface
from cognee.infrastructure.files.utils.guess_file_type import guess_file_type
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
            "advanced_pdf_loader",
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
        self,
        file_path: str,
        preferred_loaders: dict[str, dict[str, Any]],
    ) -> Optional[LoaderInterface]:
        """
        Get appropriate loader for a file.

        Args:
            file_path: Path to the file to be processed
            preferred_loaders: List of preferred loader names to try first

        Returns:
            LoaderInterface that can handle the file, or None if not found
        """
        from pathlib import Path

        file_info = guess_file_type(file_path)

        path_extension = Path(file_path).suffix.lstrip(".")

        # Try preferred loaders first
        if preferred_loaders:
            for loader_name in preferred_loaders:
                if loader_name in self._loaders:
                    loader = self._loaders[loader_name]
                    # Try with path extension first (for text formats like html)
                    if loader.can_handle(extension=path_extension, mime_type=file_info.mime):
                        return loader
                    # Fall back to content-detected extension
                    if loader.can_handle(extension=file_info.extension, mime_type=file_info.mime):
                        return loader
                else:
                    logger.info(f"Skipping {loader_name}: Preferred Loader not registered")

        # Try default priority order
        for loader_name in self.default_loader_priority:
            if loader_name in self._loaders:
                loader = self._loaders[loader_name]
                # Try with path extension first (for text formats like html)
                if loader.can_handle(extension=path_extension, mime_type=file_info.mime):
                    return loader
                # Fall back to content-detected extension
                if loader.can_handle(extension=file_info.extension, mime_type=file_info.mime):
                    return loader
            else:
                logger.info(
                    f"Skipping {loader_name}: Loader not registered (in default priority list)."
                )

        return None

    async def load_file(
        self,
        file_path: str,
        preferred_loaders: dict[str, dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Load file using appropriate loader.

        Args:
            file_path: Path to the file to be processed
            preferred_loaders: Dict of loader names to their configurations
            **kwargs: Additional loader-specific configuration

        Raises:
            ValueError: If no suitable loader is found
            Exception: If file processing fails
        """
        loader = self.get_loader(file_path, preferred_loaders)
        if not loader:
            raise ValueError(f"No loader found for file: {file_path}")

        logger.debug(f"Loading {file_path} with {loader.loader_name}")

        # Extract loader-specific config from preferred_loaders
        loader_config = {}
        if preferred_loaders and loader.loader_name in preferred_loaders:
            loader_config = preferred_loaders[loader.loader_name]

        # Merge with any additional kwargs (kwargs take precedence)
        merged_kwargs = {**loader_config, **kwargs}

        return await loader.load(file_path, **merged_kwargs)

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
