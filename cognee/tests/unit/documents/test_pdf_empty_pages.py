import importlib
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.loaders.external import pypdf_loader
from cognee.infrastructure.loaders.external.pypdf_loader import PyPdfLoader
from cognee.modules.data.processing.document_types.PdfDocument import PdfDocument

pdf_document_module = importlib.import_module(
    "cognee.modules.data.processing.document_types.PdfDocument"
)


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakePdfReader:
    def __init__(self, *_args, **_kwargs):
        self.pages = [FakePage(None), FakePage("   "), FakePage("page text")]


class CollectingChunker:
    def __init__(self, _document, get_text, max_chunk_size):
        self.get_text = get_text

    async def read(self):
        async for text in self.get_text():
            yield text


@pytest.mark.asyncio
async def test_pdf_document_skips_empty_pages(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_document_module, "PdfReader", FakePdfReader)
    file_path = tmp_path / "empty-pages.pdf"
    file_path.write_bytes(b"pdf")
    document = PdfDocument(
        id=uuid4(),
        name="empty-pages.pdf",
        raw_data_location=str(file_path),
        external_metadata="",
        mime_type="application/pdf",
    )

    texts = [text async for text in document.read(CollectingChunker, max_chunk_size=100)]

    assert texts == ["page text"]


@pytest.mark.asyncio
async def test_pypdf_loader_skips_empty_pages(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakePdfReader))

    async def fake_get_file_metadata(_file):
        return {"content_hash": "abc123"}

    monkeypatch.setattr(pypdf_loader, "get_file_metadata", fake_get_file_metadata)
    file_path = tmp_path / "empty-pages.pdf"
    file_path.write_bytes(b"pdf")

    content = await PyPdfLoader().load(str(file_path), persist=False)

    assert content == "Page 3:\npage text\n"
