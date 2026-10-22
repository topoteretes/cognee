# Adapted from topoteretes/cognee-community (Apache-2.0), commit 21d0b36.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Unit tests for Google Drive content extraction (mime-type dispatch)."""

import io
import pathlib

import pytest
from pypdf import PdfWriter

from cognee.tasks.ingestion.connectors import google_drive as gd_source
from cognee.tasks.ingestion.connectors.google_drive import (
    GOOGLE_DOC_MIME_TYPE,
    GOOGLE_SHEET_MIME_TYPE,
    PDF_MIME_TYPE,
    ExportTooLargeError,
    LoaderUnavailableError,
    extract_file_content,
    is_supported_mime_type,
)


@pytest.fixture(autouse=True)
def fake_media_stream(monkeypatch):
    """googleapiclient is an optional extra; the chunked download is replaced by a copy
    of what the fake request returns, so the tests cover the spool, decode and parse."""
    monkeypatch.setattr(
        gd_source, "_stream_media", lambda request, buffer: buffer.write(request.execute())
    )


class _FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self, num_retries=0):
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

    def get_media(self, fileId, supportsAllDrives):
        assert supportsAllDrives is True
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
    content = extract_file_content(service, "f5", "application/vnd.google-apps.drawing", "Drawing")
    assert content is None


def test_is_supported_mime_type():
    assert is_supported_mime_type(GOOGLE_DOC_MIME_TYPE)
    assert is_supported_mime_type(GOOGLE_SHEET_MIME_TYPE)
    assert is_supported_mime_type(PDF_MIME_TYPE)
    assert is_supported_mime_type("text/plain")
    assert is_supported_mime_type("application/vnd.google-apps.presentation")
    for mime in (
        "text/html",
        "text/x-python",
        "application/json",
        "application/xml",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        assert is_supported_mime_type(mime)
    for mime in (
        "application/vnd.google-apps.drawing",
        "application/vnd.google-apps.form",
        "image/png",
        "application/zip",
        "application/msword",
    ):
        assert not is_supported_mime_type(mime)


def test_drive_api_failure_skips_the_file():
    # A per-file extraction error must not abort the whole folder sync — the
    # file is skipped (None) and the sync continues with the other files.
    service = _FakeService(_FakeFiles(raise_error=True))
    content = extract_file_content(service, "f6", GOOGLE_DOC_MIME_TYPE, "Doc")
    assert content is None


def test_large_text_download_spills_to_disk_and_decodes_whole(monkeypatch):
    monkeypatch.setattr(gd_source, "DOWNLOAD_SPOOL_BYTES", 1024)
    text = "line of text\n" * 50_000
    service = _FakeService(_FakeFiles(media_result=text.encode()))
    assert extract_file_content(service, "big", "text/plain", "big.txt") == text


def test_invalid_utf8_is_replaced_not_fatal():
    service = _FakeService(_FakeFiles(media_result=b"ok \xff ok"))
    assert extract_file_content(service, "f", "text/plain", "n.txt") == "ok \ufffd ok"


def test_export_over_the_google_limit_is_named():
    class _Http403(Exception):
        class resp:
            status = 403

    class _Files:
        def export(self, fileId, mimeType):
            raise _Http403(
                '<HttpError 403 returned "This file is too large to be exported.". '
                "Details: \"[{'reason': 'exportSizeLimitExceeded'}]\">"
            )

    with pytest.raises(ExportTooLargeError):
        extract_file_content(_FakeService(_Files()), "f", GOOGLE_DOC_MIME_TYPE, "Doc")


def test_other_403_is_an_ordinary_skipped_extraction():
    class _Http403(Exception):
        class resp:
            status = 403

    class _Files:
        def export(self, fileId, mimeType):
            raise _Http403("forbidden: insufficientPermissions")

    assert extract_file_content(_FakeService(_Files()), "f", GOOGLE_DOC_MIME_TYPE, "Doc") is None


def test_google_slides_export_as_plain_text():
    seen = {}

    class _Files:
        def export(self, fileId, mimeType):
            seen["mime"] = mimeType
            return _FakeRequest(b"slide one\nslide two")

    service = _FakeService(_Files())
    assert extract_file_content(
        service, "s", "application/vnd.google-apps.presentation", "Deck"
    ) == ("slide one\nslide two")
    assert seen["mime"] == "text/plain"


class _FakeLoaderEngine:
    def __init__(self, loader_available=True, text="from the loader"):
        self.loader_available = loader_available
        self.text = text
        self.seen = {}

    def get_loader(self, file_path, preferred_loaders):
        return object() if self.loader_available else None

    def _no_loader_message(self, file_path):
        return f"install cognee[docling] for {file_path}"

    async def load_file(self, file_path, preferred_loaders=None, **kwargs):
        self.seen = {
            "suffix": pathlib.Path(file_path).suffix,
            "bytes": pathlib.Path(file_path).read_bytes(),
            "kwargs": kwargs,
        }
        return self.text


@pytest.fixture
def loader_engine(monkeypatch):
    import sys

    import cognee.infrastructure.loaders.get_loader_engine

    engine = _FakeLoaderEngine()
    monkeypatch.setattr(
        sys.modules["cognee.infrastructure.loaders.get_loader_engine"],
        "get_loader_engine",
        lambda: engine,
    )
    return engine


DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_office_file_goes_through_the_loader_engine_without_persisting(loader_engine):
    service = _FakeService(_FakeFiles(media_result=b"PK-docx-bytes"))
    assert extract_file_content(service, "d", DOCX, "report.docx") == "from the loader"
    assert loader_engine.seen == {
        "suffix": ".docx",
        "bytes": b"PK-docx-bytes",
        "kwargs": {"persist": False},
    }


def test_missing_loader_is_a_named_error_not_a_skip(loader_engine):
    loader_engine.loader_available = False
    service = _FakeService(_FakeFiles(media_result=b"x"))
    with pytest.raises(LoaderUnavailableError, match="cognee\\[docling\\]"):
        extract_file_content(service, "d", DOCX, "report.docx")


def test_broken_office_file_is_an_ordinary_skipped_extraction(loader_engine):
    async def broken(*_args, **_kwargs):
        raise RuntimeError("corrupt docx")

    loader_engine.load_file = broken
    service = _FakeService(_FakeFiles(media_result=b"x"))
    assert extract_file_content(service, "d", DOCX, "report.docx") is None


def test_missing_loader_fails_the_file_with_its_reason(loader_engine):
    loader_engine.loader_available = False
    stats = {}
    meta = {"id": "d", "name": "report.docx", "mimeType": DOCX}
    config = gd_source._DriveConfig(
        folder_id="root",
        auth_mode="service_account",
        credentials_path=None,
        token_path=None,
        include_subfolders=True,
    )
    row = gd_source._file_to_row(_FakeService(_FakeFiles(media_result=b"x")), meta, config, stats)
    assert row is None
    assert stats["failed"] == 1
    assert stats["failed_loader_unavailable"] == 1
