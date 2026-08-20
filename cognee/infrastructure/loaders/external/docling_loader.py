import os
import sys
import types
from functools import lru_cache
from typing import Any

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.loaders.store_derived_text import store_derived_text

logger = get_logger(__name__)

# ML-stack packages the slim docling profile deliberately leaves out. Which one
# a conversion trips over first shifts between docling-slim releases (2.115
# failed on docling_ibm_models, 2.118 on torch), so match the whole set.
_SLIM_OMITTED_ML_MODULES = ("docling_ibm_models", "torch", "torchvision", "transformers")


def _is_slim_omitted_module(error: ModuleNotFoundError) -> bool:
    return (error.name or "").split(".")[0] in _SLIM_OMITTED_ML_MODULES


def _install_docling_ibm_models_stubs() -> None:
    """Make ``docling.document_converter`` importable on docling-slim installs.

    docling-slim (the ``cognee[docling]`` extra) omits ``docling-ibm-models``
    because that package hard-depends on torch. Docling's import audit for slim
    installs is still incomplete upstream: ``DocumentConverter`` unconditionally
    imports two pure-Python, torch-free modules from ``docling_ibm_models`` (the
    rule-based reading order used by the PDF pipeline). Stub exactly those two
    modules so the import succeeds; the stub classes raise on instantiation,
    which only happens when the torch-based PDF/image pipeline is actually built
    — and that pipeline needs the real package (``cognee[docling-full]``) anyway.
    """

    def _unavailable(name: str) -> type:
        class _Unavailable:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ModuleNotFoundError(
                    f"{name} requires docling-ibm-models, which is not part of the slim "
                    "docling install. Install with: pip install 'cognee[docling-full]'"
                )

        _Unavailable.__name__ = name
        return _Unavailable

    pkg = types.ModuleType("docling_ibm_models")
    pkg.__path__ = []
    list_item_normalizer = types.ModuleType("docling_ibm_models.list_item_normalizer")
    list_item_normalizer.__path__ = []
    list_marker_processor = types.ModuleType(
        "docling_ibm_models.list_item_normalizer.list_marker_processor"
    )
    list_marker_processor.ListItemMarkerProcessor = _unavailable("ListItemMarkerProcessor")  # ty: ignore[unresolved-attribute]
    reading_order = types.ModuleType("docling_ibm_models.reading_order")
    reading_order.__path__ = []
    reading_order_rb = types.ModuleType("docling_ibm_models.reading_order.reading_order_rb")
    reading_order_rb.PageElement = _unavailable("PageElement")  # ty: ignore[unresolved-attribute]
    reading_order_rb.ReadingOrderPredictor = _unavailable("ReadingOrderPredictor")  # ty: ignore[unresolved-attribute]

    for module in (
        pkg,
        list_item_normalizer,
        list_marker_processor,
        reading_order,
        reading_order_rb,
    ):
        sys.modules[module.__name__] = module


def _import_document_converter():
    try:
        from docling.document_converter import DocumentConverter  # ty:ignore[unresolved-import]
    except ModuleNotFoundError as e:
        if not (e.name or "").startswith("docling_ibm_models"):
            raise
        _install_docling_ibm_models_stubs()
        from docling.document_converter import DocumentConverter  # ty:ignore[unresolved-import]

    return DocumentConverter


@lru_cache(maxsize=1)
def _get_docling_converter():
    try:
        DocumentConverter = _import_document_converter()
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
            try:
                conv_result = converter.convert(file_path)
            except ModuleNotFoundError as e:
                if not _is_slim_omitted_module(e):
                    raise
                raise ImportError(
                    f"Converting '{file_path}' with docling requires its torch-based models, "
                    "which are not part of the slim docling install. "
                    "Install with: pip install 'cognee[docling-full]'"
                ) from e
            text = conv_result.document.export_to_text()

            if not kwargs.get("persist", True):
                return text

            storage_config = get_storage_config()
            data_root_directory = storage_config["data_root_directory"]
            storage = get_file_storage(data_root_directory)

            return await store_derived_text(storage, storage_file_name, text)
