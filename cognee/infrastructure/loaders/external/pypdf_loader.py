from typing import List
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata

logger = get_logger(__name__)


class PyPdfLoader(LoaderInterface):
    """
    PDF loader using pypdf library.

    Extracts text content from PDF files page by page, providing
    structured page information and handling PDF-specific errors.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

    @property
    def supported_mime_types(self) -> List[str]:
        return ["application/pdf"]

    @property
    def loader_name(self) -> str:
        return "pypdf_loader"

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """Check if file can be handled by this loader."""
        # Check file extension
        if extension in self.supported_extensions and mime_type in self.supported_mime_types:
            return True

        return False

    async def load(self, file_path: str, strict: bool = False, **kwargs) -> str:
        """
        Load PDF file and extract text content.

        Args:
            file_path: Path to the PDF file
            strict: Whether to use strict mode for PDF reading
            **kwargs: Additional arguments

        Returns:
            LoaderResult with extracted text content and metadata

        Raises:
            ImportError: If pypdf is not installed
            Exception: If PDF processing fails
        """
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError(
                "pypdf is required for PDF processing. Install with: pip install pypdf"
            ) from e

        try:
            with open(file_path, "rb") as file:
                file_metadata = await get_file_metadata(file)
                # Name ingested file of current loader based on original file content hash
                storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"

                logger.info(f"Reading PDF: {file_path}")
                reader = PdfReader(file, strict=strict)

                content_parts = []
                page_texts = []

                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            page_texts.append(page_text)
                            content_parts.append(f"Page {page_num}:\n{page_text}\n")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {e}")
                        continue

                # Combine all content
                full_content = "\n".join(content_parts)

                storage_config = get_storage_config()
                data_root_directory = storage_config["data_root_directory"]
                storage = get_file_storage(data_root_directory)

                full_file_path = await storage.store(storage_file_name, full_content)

                return full_file_path

        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            raise Exception(f"PDF processing failed: {e}") from e
