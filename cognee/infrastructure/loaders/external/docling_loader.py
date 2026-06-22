import os
from functools import lru_cache
from typing import Any

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_docling_converter():
    try:
        from docling.document_converter import DocumentConverter  # ty:ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "docling is required for DoclingLoader. Install with: pip install 'cognee[docling]'"
        ) from e

    return DocumentConverter()


def _get_docling_supported_extensions() -> set[str]:
    """
    Pull extensions directly from Docling's format map.
    Returns extensions without leading dots (e.g. 'pdf', 'docx').
    """
    try:
        from docling.datamodel.base_models import FormatToExtensions  # ty:ignore[unresolved-import]
    except ImportError:
        return set()

    ext_set: set[str] = set()
    for extensions in FormatToExtensions.values():
        for ext in extensions:
            ext_set.add(ext.lower().lstrip("."))

    return ext_set


class DoclingLoader(LoaderInterface):
    loader_name = "docling_loader"

    @property
    def supported_extensions(self) -> list[str]:
        return sorted(_get_docling_supported_extensions())

    @property
    def supported_mime_types(self) -> list[str]:
        return ["*/*"]

    def can_handle(self, extension: str, mime_type: str) -> bool:
        if not extension:
            return False
        return extension.lower().lstrip(".") in _get_docling_supported_extensions()

    async def load(self, file_path: str, **kwargs: Any) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_metadata = await get_file_metadata(f)

            storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"

            converter = _get_docling_converter()
            conv_result = converter.convert(file_path)
            text = conv_result.document.export_to_text()

            if not kwargs.get("persist", True):
                return text

            storage_config = get_storage_config()
            data_root_directory = storage_config["data_root_directory"]
            storage = get_file_storage(data_root_directory)

            return await storage.store(storage_file_name, text)
