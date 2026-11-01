# Adapted from topoteretes/cognee-community (Apache-2.0), commit 21d0b36.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Unit tests for Google Drive content extraction (mime-type dispatch)."""

import io
import pathlib

import drive_pdfs
import pytest
from pypdf import PdfWriter

from cognee.tasks.ingestion.connectors import google_drive as gd_source
from cognee.tasks.ingestion.connectors.google_drive import (
    GOOGLE_DOC_MIME_TYPE,
    GOOGLE_SHEET_MIME_TYPE,
    PDF_MIME_TYPE,
    ExportTooLargeError,
    LoaderUnavailableError,
    UnreadableFileError,
    extract_file_content,
    is_supported_mime_type,
)

_REAL_STREAM_MEDIA = gd_source._stream_media


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
    with pytest.raises(LoaderUnavailableError, match="docling"):
        extract_file_content(service, "d", DOCX, "report.docx")


def test_a_loader_failure_is_transient_not_a_verdict_on_the_file(loader_engine):
    # unstructured re-raises everything, a model download included, as one RuntimeError.
    async def broken(*_args, **_kwargs):
        raise RuntimeError("Document processing failed: HTTP 503 fetching a model")

    loader_engine.load_file = broken
    service = _FakeService(_FakeFiles(media_result=b"x"))
    assert extract_file_content(service, "d", DOCX, "report.docx") is None


def test_the_coroutine_of_a_failed_asyncio_run_is_closed(loader_engine):
    import asyncio
    import inspect

    created = []
    real_load_file = loader_engine.load_file

    def tracking(*args, **kwargs):
        coroutine = real_load_file(*args, **kwargs)
        created.append(coroutine)
        return coroutine

    loader_engine.load_file = tracking

    async def inside_a_loop():
        service = _FakeService(_FakeFiles(media_result=b"x"))
        return extract_file_content(service, "d", DOCX, "report.docx")

    assert asyncio.run(inside_a_loop()) is None
    assert [inspect.getcoroutinestate(coroutine) for coroutine in created] == ["CORO_CLOSED"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_running_event_loop_is_transient_not_a_verdict_on_the_file(loader_engine):
    import asyncio

    async def inside_a_loop():
        service = _FakeService(_FakeFiles(media_result=b"x"))
        return extract_file_content(service, "d", DOCX, "report.docx")

    assert asyncio.run(inside_a_loop()) is None


def test_a_package_the_loader_needs_is_an_environment_problem_not_the_files(loader_engine):
    async def missing(*_args, **_kwargs):
        raise ImportError("docling needs torch")

    loader_engine.load_file = missing
    service = _FakeService(_FakeFiles(media_result=b"x"))
    with pytest.raises(LoaderUnavailableError, match="torch"):
        extract_file_content(service, "d", DOCX, "report.docx")


def test_password_protected_pdf_is_unreadable():
    service = _FakeService(_FakeFiles(media_result=drive_pdfs.locked_pdf()))
    with pytest.raises(UnreadableFileError, match="not been decrypted"):
        extract_file_content(service, "p", PDF_MIME_TYPE, "locked.pdf")


def test_a_pypdf_value_error_is_unreadable(monkeypatch):
    def refuses(_stream):
        raise ValueError("bad xref")

    monkeypatch.setattr(gd_source, "_extract_pdf_text", refuses)
    service = _FakeService(_FakeFiles(media_result=b"x"))
    with pytest.raises(UnreadableFileError, match="bad xref"):
        extract_file_content(service, "p", PDF_MIME_TYPE, "a.pdf")


@pytest.mark.parametrize("name", list(drive_pdfs.CORRUPT_PDFS))
def test_corrupt_pdf_is_unreadable_and_names_what_pypdf_raised(name):
    data = drive_pdfs.CORRUPT_PDFS[name]
    # What pypdf itself raises, to keep the case honest. The kind depends on the pypdf
    # release (6.21 turned the KeyError and AttributeError of two cases into a
    # UnicodeDecodeError and a PdfReadError); the connector names whichever it was.
    with pytest.raises(Exception) as raised:
        drive_pdfs.extract(data)
    service = _FakeService(_FakeFiles(media_result=data))
    with pytest.raises(UnreadableFileError, match=type(raised.value).__name__):
        extract_file_content(service, "p", PDF_MIME_TYPE, "broken.pdf")


def test_a_valid_hand_built_pdf_is_read():
    assert drive_pdfs.extract(drive_pdfs.VALID) == "hello"
    service = _FakeService(_FakeFiles(media_result=drive_pdfs.VALID))
    assert extract_file_content(service, "p", PDF_MIME_TYPE, "ok.pdf") == "hello"


def test_a_missing_cryptography_package_is_the_environment_not_the_file(monkeypatch):
    from pypdf.errors import DependencyError

    def needs_cryptography(_stream):
        raise DependencyError("cryptography>=3.1 is required for AES algorithm")

    monkeypatch.setattr(gd_source, "_extract_pdf_text", needs_cryptography)
    service = _FakeService(_FakeFiles(media_result=b"x"))
    assert extract_file_content(service, "p", PDF_MIME_TYPE, "aes.pdf") is None


def test_corrupt_pdf_is_unreadable():
    service = _FakeService(_FakeFiles(media_result=b"this is not a pdf at all"))
    with pytest.raises(UnreadableFileError):
        extract_file_content(service, "p", PDF_MIME_TYPE, "broken.pdf")


def test_a_failed_download_is_transient_not_unreadable(monkeypatch):
    # Not an OSError on purpose: only where it happens keeps it from being the file's fault.
    def network_down(request, buffer):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(gd_source, "_stream_media", network_down)
    service = _FakeService(_FakeFiles(media_result=b"x"))
    assert extract_file_content(service, "p", PDF_MIME_TYPE, "a.pdf") is None


@pytest.mark.parametrize("error", [MemoryError(), OSError("disk full")])
def test_resource_exhaustion_in_a_parser_is_not_the_files_fault(monkeypatch, error):
    def exhausted(_stream):
        raise error

    monkeypatch.setattr(gd_source, "_extract_pdf_text", exhausted)
    service = _FakeService(_FakeFiles(media_result=b"x"))
    assert extract_file_content(service, "p", PDF_MIME_TYPE, "a.pdf") is None


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


def test_download_is_chunked_with_retries(monkeypatch):
    import sys
    import types

    seen = {"retries": []}

    class _Downloader:
        def __init__(self, buffer, request, chunksize):
            seen["chunksize"] = chunksize
            self.buffer, self.calls = buffer, 0

        def next_chunk(self, num_retries=0):
            seen["retries"].append(num_retries)
            self.calls += 1
            self.buffer.write(b"chunk%d;" % self.calls)
            return None, self.calls == 3

    module = types.ModuleType("googleapiclient.http")
    module.MediaIoBaseDownload = _Downloader
    monkeypatch.setitem(sys.modules, "googleapiclient", types.ModuleType("googleapiclient"))
    monkeypatch.setitem(sys.modules, "googleapiclient.http", module)
    buffer = io.BytesIO()
    _REAL_STREAM_MEDIA(object(), buffer)
    assert buffer.getvalue() == b"chunk1;chunk2;chunk3;"
    assert seen["chunksize"] == gd_source.DOWNLOAD_CHUNK_BYTES
    assert seen["retries"] == [gd_source.API_RETRIES] * 3


def test_text_is_read_from_a_file_object_that_has_only_read_and_seek(monkeypatch):
    """Before Python 3.11 a SpooledTemporaryFile has no readable(), so io.TextIOWrapper fails on it."""
    from contextlib import contextmanager

    class _OnlyReadAndSeek:
        def __init__(self, data):
            self._buffer = io.BytesIO(data)

        def read(self, size=-1):
            return self._buffer.read(size)

        def seek(self, offset, whence=0):
            return self._buffer.seek(offset, whence)

    @contextmanager
    def download(_service, _file_id):
        yield _OnlyReadAndSeek("héllo \xff wörld".encode("latin-1"))

    monkeypatch.setattr(gd_source, "_download_media", download)
    service = _FakeService(_FakeFiles())
    assert (
        extract_file_content(service, "t", "text/plain", "n.txt") == "h\ufffdllo \ufffd w\ufffdrld"
    )
