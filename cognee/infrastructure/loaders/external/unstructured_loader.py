from typing import List
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata

logger = get_logger(__name__)


class UnstructuredLoader(LoaderInterface):
    """
    Document loader using the unstructured library.

    Handles various document formats including docx, pptx, xlsx, odt, etc.
    Uses the unstructured library's auto-partition functionality.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [
            "docx",
            "doc",
            "odt",  # Word documents
            "xlsx",
            "xls",
            "ods",  # Spreadsheets
            "pptx",
            "ppt",
            "odp",  # Presentations
            "rtf",
            "html",
            "htm",  # Rich text and HTML
            "eml",
            "msg",  # Email formats
            "epub",  # eBooks
        ]

    @property
    def supported_mime_types(self) -> List[str]:
        return [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
            "application/msword",  # doc
            "application/vnd.oasis.opendocument.text",  # odt
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
            "application/vnd.ms-excel",  # xls
            "application/vnd.oasis.opendocument.spreadsheet",  # ods
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx
            "application/vnd.ms-powerpoint",  # ppt
            "application/vnd.oasis.opendocument.presentation",  # odp
            "application/rtf",  # rtf
            "text/html",  # html
            "message/rfc822",  # eml
            "application/epub+zip",  # epub
        ]

    @property
    def loader_name(self) -> str:
        return "unstructured_loader"

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """Check if file can be handled by this loader."""
        # Check file extension
        if extension in self.supported_extensions and mime_type in self.supported_mime_types:
            return True

        return False

    async def load(self, file_path: str, strategy: str = "auto", **kwargs):
        """
        Load document using unstructured library.

        Args:
            file_path: Path to the document file
            strategy: Partitioning strategy ("auto", "fast", "hi_res", "ocr_only")
            **kwargs: Additional arguments passed to unstructured partition

        Returns:
            LoaderResult with extracted text content and metadata

        Raises:
            ImportError: If unstructured is not installed
            Exception: If document processing fails
        """
        try:
            from unstructured.partition.auto import partition
        except ImportError as e:
            raise ImportError(
                "unstructured is required for document processing. "
                "Install with: pip install unstructured"
            ) from e

        try:
            logger.info(f"Processing document: {file_path}")

            with open(file_path, "rb") as f:
                file_metadata = await get_file_metadata(f)
            # Name ingested file of current loader based on original file content hash
            storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"

            # Set partitioning parameters
            partition_kwargs = {"filename": file_path, "strategy": strategy, **kwargs}

            # Use partition to extract elements
            elements = partition(**partition_kwargs)

            # Process elements into text content
            text_parts = []

            for element in elements:
                element_text = str(element).strip()
                if element_text:
                    text_parts.append(element_text)

            # Combine all text content
            full_content = "\n\n".join(text_parts)

            storage_config = get_storage_config()
            data_root_directory = storage_config["data_root_directory"]
            storage = get_file_storage(data_root_directory)

            full_file_path = await storage.store(storage_file_name, full_content)

            return full_file_path

        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise Exception(f"Document processing failed: {e}") from e
