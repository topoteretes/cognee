import os
import importlib.util
from typing import Dict, List, Optional
from .LoaderInterface import LoaderInterface
from .models.LoaderResult import LoaderResult
from cognee.shared.logging_utils import get_logger


class LoaderEngine:
    """
    Main loader engine for managing file loaders.

    Follows cognee's adapter pattern similar to database engines,
    providing a centralized system for file loading operations.
    """

    def __init__(
        self,
        loader_directories: List[str],
        default_loader_priority: List[str],
        fallback_loader: str = "text_loader",
        enable_dependency_validation: bool = True,
    ):
        """
        Initialize the loader engine.

        Args:
            loader_directories: Directories to search for loader implementations
            default_loader_priority: Priority order for loader selection
            fallback_loader: Default loader to use when no other matches
            enable_dependency_validation: Whether to validate loader dependencies
        """
        self._loaders: Dict[str, LoaderInterface] = {}
        self._extension_map: Dict[str, List[LoaderInterface]] = {}
        self._mime_type_map: Dict[str, List[LoaderInterface]] = {}
        self.loader_directories = loader_directories
        self.default_loader_priority = default_loader_priority
        self.fallback_loader = fallback_loader
        self.enable_dependency_validation = enable_dependency_validation
        self.logger = get_logger(__name__)

    def register_loader(self, loader: LoaderInterface) -> bool:
        """
        Register a loader with the engine.

        Args:
            loader: LoaderInterface implementation to register

        Returns:
            True if loader was registered successfully, False otherwise
        """
        # Validate dependencies if enabled
        if self.enable_dependency_validation and not loader.validate_dependencies():
            self.logger.warning(
                f"Skipping loader '{loader.loader_name}' - missing dependencies: "
                f"{loader.get_dependencies()}"
            )
            return False

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

        self.logger.info(f"Registered loader: {loader.loader_name}")
        return True

    def get_loader(
        self, file_path: str, mime_type: str = None, preferred_loaders: List[str] = None
    ) -> Optional[LoaderInterface]:
        """
        Get appropriate loader for a file.

        Args:
            file_path: Path to the file to be processed
            mime_type: Optional MIME type of the file
            preferred_loaders: List of preferred loader names to try first

        Returns:
            LoaderInterface that can handle the file, or None if not found
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Try preferred loaders first
        if preferred_loaders:
            for loader_name in preferred_loaders:
                if loader_name in self._loaders:
                    loader = self._loaders[loader_name]
                    if loader.can_handle(file_path, mime_type):
                        return loader

        # Try priority order
        for loader_name in self.default_loader_priority:
            if loader_name in self._loaders:
                loader = self._loaders[loader_name]
                if loader.can_handle(file_path, mime_type):
                    return loader

        # Try mime type mapping
        if mime_type and mime_type in self._mime_type_map:
            for loader in self._mime_type_map[mime_type]:
                if loader.can_handle(file_path, mime_type):
                    return loader

        # Try extension mapping
        if ext in self._extension_map:
            for loader in self._extension_map[ext]:
                if loader.can_handle(file_path, mime_type):
                    return loader

        # Fallback loader
        if self.fallback_loader in self._loaders:
            fallback = self._loaders[self.fallback_loader]
            if fallback.can_handle(file_path, mime_type):
                return fallback

        return None

    async def load_file(
        self, file_path: str, mime_type: str = None, preferred_loaders: List[str] = None, **kwargs
    ) -> LoaderResult:
        """
        Load file using appropriate loader.

        Args:
            file_path: Path to the file to be processed
            mime_type: Optional MIME type of the file
            preferred_loaders: List of preferred loader names to try first
            **kwargs: Additional loader-specific configuration

        Returns:
            LoaderResult containing processed content and metadata

        Raises:
            ValueError: If no suitable loader is found
            Exception: If file processing fails
        """
        loader = self.get_loader(file_path, mime_type, preferred_loaders)
        if not loader:
            raise ValueError(f"No loader found for file: {file_path}")

        self.logger.debug(f"Loading {file_path} with {loader.loader_name}")
        return await loader.load(file_path, **kwargs)

    def discover_loaders(self):
        """
        Auto-discover loaders from configured directories.

        Scans loader directories for Python modules containing
        LoaderInterface implementations and registers them.
        """
        for directory in self.loader_directories:
            if os.path.exists(directory):
                self._discover_in_directory(directory)

    def _discover_in_directory(self, directory: str):
        """
        Discover loaders in a specific directory.

        Args:
            directory: Directory path to scan for loader implementations
        """
        try:
            for file_name in os.listdir(directory):
                if file_name.endswith(".py") and not file_name.startswith("_"):
                    module_name = file_name[:-3]
                    file_path = os.path.join(directory, file_name)

                    try:
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                            # Look for loader classes
                            for attr_name in dir(module):
                                attr = getattr(module, attr_name)
                                if (
                                    isinstance(attr, type)
                                    and issubclass(attr, LoaderInterface)
                                    and attr != LoaderInterface
                                ):
                                    # Instantiate and register the loader
                                    try:
                                        loader_instance = attr()
                                        self.register_loader(loader_instance)
                                    except Exception as e:
                                        self.logger.warning(
                                            f"Failed to instantiate loader {attr_name}: {e}"
                                        )

                    except Exception as e:
                        self.logger.warning(f"Failed to load module {module_name}: {e}")

        except OSError as e:
            self.logger.warning(f"Failed to scan directory {directory}: {e}")

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
            "dependencies": loader.get_dependencies(),
            "available": loader.validate_dependencies(),
        }
