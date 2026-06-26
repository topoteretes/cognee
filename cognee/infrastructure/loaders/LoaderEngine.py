from pathlib import Path
from typing import Any, BinaryIO

from cognee.infrastructure.files.utils.guess_file_type import guess_file_type
from cognee.shared.logging_utils import get_logger

from .LoaderInterface import LoaderInterface

logger = get_logger(__name__)


class LoaderEngine:
    """
    Main loader engine for managing file loaders.

    Follows cognee's adapter pattern similar to database engines,
    providing a centralized system for file loading operations.
    """

    def __init__(self) -> None:
        """
        Initialize the loader engine.

        Args:
            default_loader_priority: Priority order for loader selection
        """
        self._loaders: dict[str, LoaderInterface] = {}
        self._extension_map: dict[str, list[LoaderInterface]] = {}
        self._mime_type_map: dict[str, list[LoaderInterface]] = {}

        self.default_loader_priority = [
            "text_loader",
            "pypdf_loader",
            "image_loader",
            "audio_loader",
            "csv_loader",
            "unstructured_loader",
            "advanced_pdf_loader",
            "docling_loader",
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
        preferred_loaders: dict[str, dict[str, Any]] | None,
    ) -> LoaderInterface | None:
        """
        Get appropriate loader for a file.

        Args:
            file_path: Path to the file to be processed
            preferred_loaders: List of preferred loader names to try first

        Returns:
            LoaderInterface that can handle the file, or None if not found
        """
        file_info = guess_file_type(file_path)  # ty:ignore[invalid-argument-type]

        path_extension = Path(file_path).suffix.lstrip(".")

        return self._get_loader(path_extension, file_info.extension, file_info.mime, preferred_loaders)

    def get_loader_for_stream(
        self,
        file: BinaryIO,
        file_name: str,
        preferred_loaders: dict[str, dict[str, Any]] | None,
    ) -> LoaderInterface | None:
        file_info = guess_file_type(file, file_name)
        file.seek(0)

        path_extension = Path(file_name).suffix.lstrip(".")

        return self._get_loader(path_extension, file_info.extension, file_info.mime, preferred_loaders)

    def _get_loader(
        self,
        path_extension: str,
        detected_extension: str,
        mime_type: str,
        preferred_loaders: dict[str, dict[str, Any]] | None,
    ) -> LoaderInterface | None:

        # Try preferred loaders first
        if preferred_loaders:
            for loader_name in preferred_loaders:
                if loader_name in self._loaders:
                    loader = self._loaders[loader_name]
                    # Try with path extension first (for text formats like html)
                    if loader.can_handle(extension=path_extension, mime_type=mime_type):
                        return loader
                    # Fall back to content-detected extension
                    if loader.can_handle(extension=detected_extension, mime_type=mime_type):
                        return loader
                else:
                    logger.info(f"Skipping {loader_name}: Preferred Loader not registered")

        # Try default priority order
        for loader_name in self.default_loader_priority:
            if loader_name in self._loaders:
                loader = self._loaders[loader_name]
                # Try with path extension first (for text formats like html)
                if loader.can_handle(extension=path_extension, mime_type=mime_type):
                    return loader
                # Fall back to content-detected extension
                if loader.can_handle(extension=detected_extension, mime_type=mime_type):
                    return loader
            else:
                logger.info(
                    f"Skipping {loader_name}: Loader not registered (in default priority list)."
                )

        return None

    async def load_file_stream(
        self,
        file: BinaryIO,
        file_name: str,
        preferred_loaders: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[str, LoaderInterface]:
        loader = self.get_loader_for_stream(file, file_name, preferred_loaders)
        if not loader:
            raise ValueError(self._no_loader_message(file_name))

        logger.debug(f"Loading {file_name} with {loader.loader_name}")

        loader_config = {}
        if preferred_loaders and loader.loader_name in preferred_loaders:
            loader_config = preferred_loaders[loader.loader_name]

        merged_kwargs = {**loader_config, **kwargs, "file_stream": file}

        return await loader.load(file_name, **merged_kwargs), loader

    async def load_file(
        self,
        file_path: str,
        preferred_loaders: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
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
            raise ValueError(self._no_loader_message(file_path))

        logger.debug(f"Loading {file_path} with {loader.loader_name}")

        # Extract loader-specific config from preferred_loaders
        loader_config = {}
        if preferred_loaders and loader.loader_name in preferred_loaders:
            loader_config = preferred_loaders[loader.loader_name]

        # Merge with any additional kwargs (kwargs take precedence)
        merged_kwargs = {**loader_config, **kwargs}

        return await loader.load(file_path, **merged_kwargs)

    # Extensions handled only by optional document loaders, mapped to the
    # cognee extra that provides the loader. Used to turn an opaque
    # "no loader found" into an actionable install hint.
    _OPTIONAL_FORMAT_EXTRAS = {
        "pptx": "docling",
        "ppt": "docling",
        "odp": "docling",
        "docx": "docling",
        "doc": "docling",
        "odt": "docling",
        "xlsx": "docling",
        "xls": "docling",
        "ods": "docling",
        "rtf": "docling",
        "html": "docling",
        "htm": "docling",
        "eml": "docling",
        "msg": "docling",
        "epub": "docling",
    }

    def _no_loader_message(self, file_path: str) -> str:
        """Build an actionable error for a file no registered loader can handle.

        Names the extension, lists the currently supported extensions, and — for
        office/document formats that only ship via an optional loader — tells the
        user which extra to install (e.g. ``.pptx`` needs ``cognee[docling]``).
        """
        from pathlib import Path

        ext = Path(file_path).suffix.lstrip(".").lower()
        supported = ", ".join(sorted(self._extension_map.keys())) or "none"

        message = f"No loader found for file '{file_path}'"
        if ext:
            message += f" (extension '.{ext}')"
        message += "."

        extra = self._OPTIONAL_FORMAT_EXTRAS.get(ext)
        if extra:
            message += (
                f" '.{ext}' files need an optional document loader that is not installed. "
                f"Install it with `pip install cognee[{extra}]` (or `cognee[unstructured]`) and retry."
            )

        message += f" Supported extensions: {supported}."
        return message

    def get_available_loaders(self) -> list[str]:
        """
        Get list of available loader names.

        Returns:
            List of registered loader names
        """
        return list(self._loaders.keys())

    def get_loader_info(self, loader_name: str) -> dict[str, Any]:
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
