"""HTML text is extracted with LF line endings, as HTML5 parsers and the Rust SDK do."""

import hashlib

import pytest

pytest.importorskip("bs4")
pytest.importorskip("lxml")

from cognee.infrastructure.loaders.external.beautiful_soup_loader import (
    BeautifulSoupLoader,
)


async def _extract(tmp_path, raw: bytes, **kwargs) -> str:
    """Write ``raw`` to an .html file and return the loader's text, not stored."""
    path = tmp_path / "page.html"
    path.write_bytes(raw)
    return await BeautifulSoupLoader().load(str(path), persist=False, **kwargs)


def _store_under(monkeypatch, root) -> None:
    """Make the loader store its derived text under ``root``."""
    monkeypatch.setattr(
        "cognee.infrastructure.files.storage.get_storage_config",
        lambda: {"data_root_directory": str(root)},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["one\r\ntwo\r\nthree", "one\rtwo\rthree"], ids=["crlf", "cr"])
async def test_default_rules_extract_lf_text(tmp_path, body):
    text = await _extract(tmp_path, f"<html><body><p>{body}</p></body></html>".encode())

    assert text == "one\ntwo\nthree"


@pytest.mark.asyncio
async def test_utf16_html_keeps_its_letters(tmp_path):
    # In UTF-16 the byte 0x0D is also part of other characters ("č" is 0D 01),
    # so replacing it in the raw bytes would corrupt the text.
    raw = "<html><body><p>one\r\nčaj žé</p></body></html>".encode("utf-16")

    assert await _extract(tmp_path, raw) == "one\nčaj žé"


@pytest.mark.asyncio
async def test_xpath_and_css_rules_give_the_same_text(tmp_path):
    raw = b"<html><body><p>one\r\ntwo\rthree</p></body></html>"

    css = await _extract(tmp_path, raw, extraction_rules={"p": "p"})
    xpath = await _extract(tmp_path, raw, extraction_rules={"p": {"xpath": "//p"}})

    assert css == xpath == "one\ntwo\nthree"


@pytest.mark.asyncio
async def test_xpath_rules_decode_like_css_rules(tmp_path):
    # lxml alone guesses Latin-1 for UTF-8 with no <meta charset>.
    raw = "<html><body><p>čaj\r\nžé</p></body></html>".encode()

    xpath = await _extract(tmp_path, raw, extraction_rules={"p": {"xpath": "//p"}})

    assert xpath == "čaj\nžé"


@pytest.mark.asyncio
async def test_carriage_return_reference_is_kept(tmp_path):
    # HTML5 parsers translate only literal line breaks; "&#13;" stays "\r".
    raw = b"<html><body><p>a&#13;b\r\nc</p></body></html>"

    css = await _extract(tmp_path, raw, extraction_rules={"p": "p"})
    xpath = await _extract(tmp_path, raw, extraction_rules={"p": {"xpath": "//p"}})

    assert css == xpath == "a\rb\nc"


@pytest.mark.asyncio
async def test_input_with_no_tags_keeps_carriage_returns(tmp_path):
    # The plain-text fallback keeps the text as it is, in both SDKs.
    assert await _extract(tmp_path, b"one\r\ntwo") == "one\r\ntwo"


@pytest.mark.asyncio
async def test_lf_text_is_stored_under_the_original_content_hash(tmp_path, monkeypatch):
    _store_under(monkeypatch, tmp_path / "data")
    raw = b"<html><body><p>one\r\ntwo</p></body></html>"
    (tmp_path / "page.html").write_bytes(raw)

    result = await BeautifulSoupLoader().load(str(tmp_path / "page.html"))

    stored = tmp_path / "data" / f"text_{hashlib.md5(raw).hexdigest()}.txt"
    assert stored.read_bytes() == b"one\ntwo"
    # Ingestion saves this as the row's raw_content_hash.
    assert result.file_metadata["content_hash"] == hashlib.md5(b"one\ntwo").hexdigest()
