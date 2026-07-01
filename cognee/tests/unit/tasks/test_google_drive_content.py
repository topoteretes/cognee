"""Unit tests for Google Drive content extraction (mime-type dispatch)."""

import io
import sys
import types

import pytest
from pypdf import PdfWriter

from cognee.tasks.ingestion.connectors.google_drive.content import (
    extract_file_content,
    is_supported_mime_type,
    GOOGLE_DOC_MIME_TYPE,
    GOOGLE_SHEET_MIME_TYPE,
    PDF_MIME_TYPE,
)
from cognee.tasks.ingestion.connectors.google_drive.exceptions import GoogleDriveAPIError
from cognee.modules.data.exceptions import UnstructuredLibraryImportError

PPTX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


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


def _make_pdf_bytes(text_pages):
    # pypdf can't easily synthesize extractable text pages, so we only
    # assert the PDF branch decodes bytes -> str without raising; real
    # text extraction is exercised by pypdf itself, already covered
    # elsewhere in the codebase (PdfDocument).
    writer = PdfWriter()
    for _ in text_pages:
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
    pdf_bytes = _make_pdf_bytes([1, 2])
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


def test_drive_api_failure_raises_google_drive_api_error():
    service = _FakeService(_FakeFiles(raise_error=True))
    with pytest.raises(GoogleDriveAPIError):
        extract_file_content(service, "f6", GOOGLE_DOC_MIME_TYPE, "Doc")


def test_pptx_is_supported_mime_type():
    assert is_supported_mime_type(PPTX_MIME_TYPE)


def test_pptx_without_unstructured_installed_raises_clear_error(monkeypatch):
    # Force the import to fail regardless of whether 'unstructured' happens
    # to be installed in the environment running this test — a `None` entry
    # in sys.modules makes Python raise ImportError for that module.
    monkeypatch.setitem(sys.modules, "unstructured.partition.auto", None)
    service = _FakeService(_FakeFiles(media_result=b"fake pptx bytes"))
    with pytest.raises(UnstructuredLibraryImportError, match="cognee\\[docs\\]"):
        extract_file_content(service, "f7", PPTX_MIME_TYPE, "Lecture_1.pptx")


def test_pptx_extracts_text_when_unstructured_available(monkeypatch):
    class _FakeElement:
        def __str__(self):
            return "Slide 1 title"

    def fake_partition(file, content_type):
        return [_FakeElement(), _FakeElement()]

    fake_auto_module = types.ModuleType("unstructured.partition.auto")
    fake_auto_module.partition = fake_partition
    fake_partition_module = types.ModuleType("unstructured.partition")
    fake_partition_module.auto = fake_auto_module
    fake_unstructured_module = types.ModuleType("unstructured")
    fake_unstructured_module.partition = fake_partition_module

    monkeypatch.setitem(sys.modules, "unstructured", fake_unstructured_module)
    monkeypatch.setitem(sys.modules, "unstructured.partition", fake_partition_module)
    monkeypatch.setitem(sys.modules, "unstructured.partition.auto", fake_auto_module)

    service = _FakeService(_FakeFiles(media_result=b"fake pptx bytes"))
    content = extract_file_content(service, "f8", PPTX_MIME_TYPE, "Lecture_1.pptx")

    assert content == "Slide 1 title\n\nSlide 1 title"
