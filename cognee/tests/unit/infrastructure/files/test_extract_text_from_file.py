from io import BytesIO
from types import SimpleNamespace

import cognee.infrastructure.files.utils.extract_text_from_file as extract_text_module
from cognee.infrastructure.files.utils.extract_text_from_file import extract_text_from_file


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakePdfReader:
    def __init__(self, stream):
        self.pages = [FakePage(None), FakePage("  extracted text  ")]


def test_extract_text_from_pdf_skips_pages_without_text(monkeypatch):
    monkeypatch.setattr(extract_text_module, "PdfReader", FakePdfReader)

    text = extract_text_from_file(BytesIO(b"pdf"), SimpleNamespace(extension="pdf"))

    assert text == "extracted text"
