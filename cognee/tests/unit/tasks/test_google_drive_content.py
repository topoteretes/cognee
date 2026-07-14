"""Unit tests for Google Drive content extraction (mime-type dispatch)."""

import io

from pypdf import PdfWriter

from cognee.tasks.ingestion.connectors.google_drive import (
    extract_file_content,
    is_supported_mime_type,
    GOOGLE_DOC_MIME_TYPE,
    GOOGLE_SHEET_MIME_TYPE,
    PDF_MIME_TYPE,
)


class _FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeFiles:
    def __init__(self, export_result=None, media_result=None, raise_error=False):
        self._export_result = export_result
        self._media_result = media_result
        self._raise_error = raise_error

    def export(self, fileId, mimeType):
        if self._raise_error:
            raise RuntimeError("export failed")
        return _FakeRequest(self._export_result)

    def get_media(self, fileId):
        if self._raise_error:
            raise RuntimeError("download failed")
        return _FakeRequest(self._media_result)


class _FakeService:
    def __init__(self, files_resource):
        self._files_resource = files_resource

    def files(self):
        return self._files_resource


def _make_pdf_bytes(page_count):
    # pypdf can't easily synthesize extractable text pages, so we only assert
    # the PDF branch decodes bytes -> str without raising; real text extraction
    # is exercised by pypdf itself, already covered elsewhere (PdfDocument).
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_google_doc_exports_as_plain_text():
    service = _FakeService(_FakeFiles(export_result=b"hello doc"))
    content = extract_file_content(service, "f1", GOOGLE_DOC_MIME_TYPE, "Doc")
    assert content == "hello doc"


def test_google_sheet_exports_as_csv():
    service = _FakeService(_FakeFiles(export_result=b"a,b\n1,2"))
    content = extract_file_content(service, "f2", GOOGLE_SHEET_MIME_TYPE, "Sheet")
    assert content == "a,b\n1,2"


def test_plain_text_file_is_downloaded_and_decoded():
    service = _FakeService(_FakeFiles(media_result=b"raw text content"))
    content = extract_file_content(service, "f3", "text/plain", "Notes")
    assert content == "raw text content"


def test_pdf_is_downloaded_and_text_extracted_without_raising():
    pdf_bytes = _make_pdf_bytes(2)
    service = _FakeService(_FakeFiles(media_result=pdf_bytes))
    content = extract_file_content(service, "f4", PDF_MIME_TYPE, "Doc.pdf")
    assert isinstance(content, str)


def test_unsupported_mime_type_returns_none():
    service = _FakeService(_FakeFiles())
    content = extract_file_content(
        service, "f5", "application/vnd.google-apps.presentation", "Slides"
    )
    assert content is None


def test_is_supported_mime_type():
    assert is_supported_mime_type(GOOGLE_DOC_MIME_TYPE)
    assert is_supported_mime_type(GOOGLE_SHEET_MIME_TYPE)
    assert is_supported_mime_type(PDF_MIME_TYPE)
    assert is_supported_mime_type("text/plain")
    assert not is_supported_mime_type("application/vnd.google-apps.presentation")


def test_drive_api_failure_skips_the_file():
    # A per-file extraction error must not abort the whole folder sync — the
    # file is skipped (None) and the sync continues with the other files.
    service = _FakeService(_FakeFiles(raise_error=True))
    content = extract_file_content(service, "f6", GOOGLE_DOC_MIME_TYPE, "Doc")
    assert content is None
