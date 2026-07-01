"""Extract plain-text content from Google Drive files, keyed by mime type."""

import io
from typing import Optional

from cognee.shared.logging_utils import get_logger
from cognee.modules.data.exceptions import UnstructuredLibraryImportError
from .exceptions import GoogleDriveAPIError

logger = get_logger("google_drive_content")

GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
GOOGLE_SHEET_MIME_TYPE = "application/vnd.google-apps.spreadsheet"
GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
PDF_MIME_TYPE = "application/pdf"
PLAIN_TEXT_MIME_TYPES = {"text/plain", "text/markdown", "text/csv"}

# Uploaded (non-Google-native) Office documents, extracted via the same
# `unstructured` library cognee's normal add() pipeline already uses for
# these formats (UnstructuredDocument). Requires `pip install cognee[docs]`
# — not bundled with the `google-drive` extra since it's a heavy optional
# dependency only needed if your Drive folder actually contains these.
OFFICE_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/msword",  # .doc
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",  # .xls
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.ms-powerpoint",  # .ppt
    "application/vnd.oasis.opendocument.text",  # .odt
    "application/vnd.oasis.opendocument.spreadsheet",  # .ods
    "application/vnd.oasis.opendocument.presentation",  # .odp
}

# Native Google formats we know how to export as text. Other native types
# (Slides, Drawings, Forms, ...) are out of scope for the MVP — skipped with
# a warning rather than failing the whole sync.
_EXPORT_MIME_MAP = {
    GOOGLE_DOC_MIME_TYPE: "text/plain",
    # Drive's export API only returns the first sheet as CSV — documented
    # limitation; multi-sheet extraction would require the Sheets API.
    GOOGLE_SHEET_MIME_TYPE: "text/csv",
}


def is_supported_mime_type(mime_type: str) -> bool:
    return (
        mime_type in _EXPORT_MIME_MAP
        or mime_type == PDF_MIME_TYPE
        or mime_type in PLAIN_TEXT_MIME_TYPES
        or mime_type in OFFICE_MIME_TYPES
    )


def extract_file_content(service, file_id: str, mime_type: str, name: str) -> Optional[str]:
    """Return extracted text content for a Drive file, or None if unsupported."""
    try:
        if mime_type in _EXPORT_MIME_MAP:
            data = (
                service.files()
                .export(fileId=file_id, mimeType=_EXPORT_MIME_MAP[mime_type])
                .execute()
            )
            return _decode(data)

        if mime_type == PDF_MIME_TYPE:
            data = service.files().get_media(fileId=file_id).execute()
            return _extract_pdf_text(data)

        if mime_type in PLAIN_TEXT_MIME_TYPES:
            data = service.files().get_media(fileId=file_id).execute()
            return _decode(data)

        if mime_type in OFFICE_MIME_TYPES:
            data = service.files().get_media(fileId=file_id).execute()
            return _extract_unstructured_text(data, mime_type)

    except (GoogleDriveAPIError, UnstructuredLibraryImportError):
        raise
    except Exception as e:
        raise GoogleDriveAPIError(
            message=f"Failed to extract content for Drive file '{name}' ({file_id}): {e}"
        ) from e

    logger.warning(
        "Skipping unsupported Drive file '%s' (%s): mime type '%s' is not supported.",
        name,
        file_id,
        mime_type,
    )
    return None


def _decode(data) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data), strict=False)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_unstructured_text(data: bytes, mime_type: str) -> str:
    """Extract text from an uploaded Office document (docx/xlsx/pptx/odt/...).

    Uses the same `unstructured` library as cognee's normal UnstructuredDocument
    pipeline. Requires the optional `docs` extra: pip install cognee[docs]
    """
    try:
        from unstructured.partition.auto import partition
    except ModuleNotFoundError:
        raise UnstructuredLibraryImportError(
            message=(
                f"Extracting content from mime type '{mime_type}' requires the "
                "'unstructured' library. Install it with: pip install cognee[docs]"
            )
        )

    elements = partition(file=io.BytesIO(data), content_type=mime_type)
    return "\n\n".join(str(el) for el in elements)
