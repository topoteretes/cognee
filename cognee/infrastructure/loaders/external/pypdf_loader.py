import os
from typing import List, Tuple
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger

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

    async def load(self, file_path: str, strict: bool = False, **kwargs) -> Tuple[str, dict]:
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

                # Gather metadata
                metadata = {
                    "name": os.path.basename(file_path),
                    "size": os.path.getsize(file_path),
                    "extension": "pdf",
                    "pages": len(reader.pages),
                    "pages_with_text": len(page_texts),
                    "loader": self.loader_name,
                }

                # Add PDF metadata if available
                if reader.metadata:
                    metadata["pdf_metadata"] = {
                        "title": reader.metadata.get("/Title", ""),
                        "author": reader.metadata.get("/Author", ""),
                        "subject": reader.metadata.get("/Subject", ""),
                        "creator": reader.metadata.get("/Creator", ""),
                        "producer": reader.metadata.get("/Producer", ""),
                        "creation_date": str(reader.metadata.get("/CreationDate", "")),
                        "modification_date": str(reader.metadata.get("/ModDate", "")),
                    }

                return full_content, metadata

        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            raise Exception(f"PDF processing failed: {e}") from e
