"""TextLoader reads a byte-order-marked file in the encoding its BOM names (SDK-1722).

Windows Notepad's "Unicode" save writes UTF-16 with a BOM. TextLoader used to open
every file as strict UTF-8, so such a file was refused with a 415 even though
presort (#6641) reads it as text. The BOM table is the one in
``decode_text_sample``, shared with presort's ``is_text_content``.
"""

import codecs

import pytest

from cognee.infrastructure.loaders.core.text_loader import TextLoader
from cognee.infrastructure.loaders.exceptions import UnreadableFileContentError
from cognee.infrastructure.loaders.get_loader_engine import get_loader_engine

TEXT = "Contact: jane.doe@example.com\nZürich, 東京\n"

BOM_ENCODINGS = [
    pytest.param(codecs.BOM_UTF16_LE, "utf-16-le", id="utf-16-le"),
    pytest.param(codecs.BOM_UTF16_BE, "utf-16-be", id="utf-16-be"),
    pytest.param(codecs.BOM_UTF32_LE, "utf-32-le", id="utf-32-le"),
    pytest.param(codecs.BOM_UTF32_BE, "utf-32-be", id="utf-32-be"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("bom, codec", BOM_ENCODINGS)
async def test_bom_file_loads_as_its_utf8_twin(tmp_path, bom, codec):
    marked = tmp_path / "contact.txt"
    marked.write_bytes(bom + TEXT.encode(codec))
    twin = tmp_path / "contact_utf8.txt"
    twin.write_text(TEXT, encoding="utf-8")

    content = await TextLoader().load(str(marked), persist=False)

    assert content == await TextLoader().load(str(twin), persist=False)
    assert content == TEXT


@pytest.mark.asyncio
async def test_utf8_bom_is_kept_as_before(tmp_path):
    """Backward compatibility: a UTF-8 BOM file loads exactly as it did before."""
    marked = tmp_path / "contact.txt"
    marked.write_bytes(codecs.BOM_UTF8 + TEXT.encode("utf-8"))

    assert await TextLoader().load(str(marked), persist=False) == "\ufeff" + TEXT


@pytest.mark.asyncio
async def test_bom_file_gets_the_same_newline_handling(tmp_path):
    marked = tmp_path / "windows.txt"
    marked.write_bytes("line one\r\nline two\r\n".encode("utf-16"))

    assert await TextLoader().load(str(marked), persist=False) == "line one\nline two\n"


@pytest.mark.asyncio
async def test_explicit_encoding_wins_over_the_bom(tmp_path):
    marked = tmp_path / "contact.txt"
    marked.write_bytes(TEXT.encode("utf-16"))

    with pytest.raises(UnreadableFileContentError) as refused:
        await TextLoader().load(str(marked), encoding="utf-8", persist=False)

    assert refused.value.status_code == 415


@pytest.mark.asyncio
async def test_empty_encoding_is_refused_not_auto_detected(tmp_path):
    """Only an omitted encoding (None) auto-detects; a blank one is a bad setting."""
    marked = tmp_path / "contact.txt"
    marked.write_bytes(TEXT.encode("utf-16"))

    with pytest.raises(LookupError):
        await TextLoader().load(str(marked), encoding="", persist=False)


@pytest.mark.asyncio
async def test_bomless_utf16_is_still_refused(tmp_path):
    """Out of scope, the same boundary as presort: without a BOM UTF-16 is ambiguous."""
    unmarked = tmp_path / "contact.txt"
    unmarked.write_bytes(TEXT.encode("utf-16-le"))

    with pytest.raises(UnreadableFileContentError) as refused:
        await TextLoader().load(str(unmarked), persist=False)

    assert refused.value.status_code == 415


@pytest.mark.asyncio
async def test_bom_prefixed_binary_that_is_valid_utf16_loads_as_text(tmp_path):
    """Accepted limitation: a BOM alone can't tell text from binary.

    Binary whose bytes after a UTF-16 BOM all form valid UTF-16 loads as control
    characters instead of being refused. Random binary almost never does (it soon
    hits an unpaired surrogate and gets the 415), and real binary formats get a
    non-text mime before reaching this loader. Presort's ``is_text_content``
    admits the same files.
    """
    blob = tmp_path / "blob.txt"
    blob.write_bytes(codecs.BOM_UTF16_LE + bytes([0x01, 0x00, 0x02, 0x00]) * 256)

    assert await TextLoader().load(str(blob), persist=False) == "\x01\x02" * 256


@pytest.mark.asyncio
async def test_bom_prefixed_random_binary_is_still_refused(tmp_path):
    blob = tmp_path / "blob.txt"
    # 0xD800 is an unpaired UTF-16 surrogate: strict decoding rejects it.
    blob.write_bytes(codecs.BOM_UTF16_LE + b"\x00\xd8\x41\x00" * 256)

    with pytest.raises(UnreadableFileContentError) as refused:
        await TextLoader().load(str(blob), persist=False)

    assert refused.value.status_code == 415


@pytest.mark.asyncio
async def test_ingestion_picks_text_loader_for_a_utf16_txt(tmp_path):
    """The loader engine is the ingestion boundary: it sniffs the file and picks the loader."""
    marked = tmp_path / "contact_utf16.txt"
    marked.write_text(TEXT, encoding="utf-16")
    engine = get_loader_engine()

    assert engine.get_loader(str(marked), None).loader_name == "text_loader"
    assert await engine.load_file(str(marked), persist=False) == TEXT
