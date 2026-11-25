"""BOM-marked UTF-16/UTF-32 text files ingest as the text they are.

An editor's "Unicode" save writes UTF-16 (or UTF-32) with a byte order mark at
the start. A strict UTF-8 read raises ``UnicodeDecodeError`` on the interleaved
null bytes, so before the BOM-aware decode such a file was refused at ingest
even though its content is plain text — while the presort scan already read it
as text. Detection and ingestion now agree through the one BOM table.
"""

import codecs

import pytest

from cognee.infrastructure.files.utils.decode_text_file import decode_text_file
from cognee.infrastructure.loaders.core.text_loader import TextLoader
from cognee.infrastructure.loaders.exceptions import UnreadableFileContentError

TEXT = "Marie Curie was born in Warsaw.\n"


@pytest.mark.parametrize(
    "bom,codec",
    [
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
        (codecs.BOM_UTF8, "utf-8"),
    ],
)
@pytest.mark.asyncio
async def test_bom_marked_text_loads_as_text(tmp_path, bom, codec):
    note = tmp_path / "note.txt"
    note.write_bytes(bom + TEXT.encode(codec))

    content = await TextLoader().load(str(note), persist=False)

    assert content == TEXT


@pytest.mark.asyncio
async def test_bom_bytes_do_not_leak_into_the_content(tmp_path):
    """A stripped BOM must not reach the caller as a stray \\ufeff."""
    note = tmp_path / "note.txt"
    note.write_bytes(codecs.BOM_UTF8 + TEXT.encode("utf-8"))

    content = await TextLoader().load(str(note), persist=False)

    assert not content.startswith("\ufeff")


@pytest.mark.asyncio
async def test_explicit_encoding_still_loads_bom_less_utf16(tmp_path):
    note = tmp_path / "note.txt"
    note.write_bytes(TEXT.encode("utf-16-le"))

    content = await TextLoader().load(str(note), encoding="utf-16-le", persist=False)

    assert content == TEXT


@pytest.mark.asyncio
async def test_binary_content_still_refused(tmp_path):
    """The decode is strict: bytes that are not text in any encoding still raise."""
    blob = tmp_path / "Installer.dmg"
    blob.write_bytes(b"x\xda" + bytes(range(256)) * 8)

    with pytest.raises(UnreadableFileContentError):
        await TextLoader().load(str(blob), persist=False)


def test_decode_text_file_defaults_to_utf8_without_a_bom():
    assert decode_text_file(TEXT.encode("utf-8")) == TEXT


def test_utf16le_leading_nul_is_not_refused_as_invalid_utf32le():
    text = "\0" + TEXT
    raw = codecs.BOM_UTF16_LE + text.encode("utf-16-le")

    assert decode_text_file(raw) == text


def test_decode_text_file_raises_on_invalid_bytes():
    with pytest.raises(UnicodeDecodeError):
        decode_text_file(b"x\xda" + bytes(range(256)) * 8)
