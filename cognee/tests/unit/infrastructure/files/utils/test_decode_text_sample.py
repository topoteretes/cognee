import pytest

from cognee.infrastructure.files.utils.decode_text_sample import (
    bom_encoding,
    decode_text_sample,
)
from cognee.infrastructure.files.utils.is_text_content import is_text_content

TEXT = "Contact: jane.doe@example.com\n"


@pytest.mark.parametrize(
    "encoding",
    ["utf-8", "utf-8-sig", "utf-16", "utf-32"],
)
def test_bom_encodings_round_trip_to_real_text(encoding):
    """A BOM-written file decodes to its text, not null-laden bytes.

    ``utf-16``/``utf-32`` without an endianness are exactly what an editor's
    "Unicode" save produces: a BOM followed by the text. The recovered string
    must contain the email so a regex scan can find it.
    """
    decoded = decode_text_sample(TEXT.encode(encoding))
    assert decoded == TEXT
    assert "jane.doe@example.com" in decoded


def test_default_is_utf8_when_no_bom():
    assert decode_text_sample(TEXT.encode("utf-8")) == TEXT


@pytest.mark.parametrize(
    "encoding, codec",
    [("utf-8-sig", "utf-8"), ("utf-16", "utf-16-le"), ("utf-32", "utf-32-le")],
)
def test_bom_encoding_names_the_codec(encoding, codec):
    assert bom_encoding(TEXT.encode(encoding)) == codec


def test_bom_encoding_is_none_without_a_bom():
    assert bom_encoding(TEXT.encode("utf-8")) is None


def test_utf32_le_is_not_read_as_utf16():
    """The UTF-16 LE BOM is a prefix of the UTF-32 LE BOM; longest-first order
    keeps a UTF-32 file from being decoded two bytes at a time."""
    decoded = decode_text_sample(TEXT.encode("utf-32"))
    assert "\x00" not in decoded


def test_truncated_multibyte_sample_does_not_raise():
    """A bounded read can cut a character in half; errors='replace' tolerates it."""
    raw = TEXT.encode("utf-16")
    decoded = decode_text_sample(raw[:-1])  # drop one byte of the last unit
    assert "jane.doe@example.com" in decoded


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_is_text_content_admits_bom_text(encoding):
    """The null bytes of UTF-16/UTF-32 must not get the file rejected as binary."""
    assert is_text_content(TEXT.encode(encoding)) is True


def test_is_text_content_still_rejects_binary():
    assert is_text_content(b"\x89PNG\r\n\x1a\n\x00\x00binary\x00data") is False


@pytest.mark.parametrize(
    "bom",
    [b"\xff\xfe", b"\xff\xfe\x00\x00", b"\xfe\xff", b"\x00\x00\xfe\xff"],
)
def test_is_text_content_admits_bom_prefixed_binary(bom):
    """A binary payload that happens to start with a BOM is admitted as text.

    Checking the BOM first means ``is_text_content`` cannot tell a real
    UTF-16/UTF-32 text file from a binary blob whose leading bytes happen to be
    ``ff fe`` / ``00 00 fe ff``: both are admitted as text. This is an accepted
    limitation, not a leak — the companion
    ``test_detect_pii_bom_prefixed_binary_finds_no_pii`` shows the regex scan
    such a blob reaches finds nothing in it. Pinned so a future tightening of
    the heuristic is a deliberate, tested change.
    """
    blob = bom + bytes(range(256)) * 8
    assert is_text_content(blob) is True


def test_is_text_content_rejects_bomless_utf16():
    """BOM-less UTF-16 stays out of scope: without a BOM its null bytes read as
    binary, so it is skipped (silently, ``warnings=[]``). Sniffing a BOM-less
    encoding is ambiguous, and ingestion's ``TextLoader`` already refuses it
    unless ``encoding`` is passed explicitly."""
    assert is_text_content("Contact: jane.doe@example.com\n".encode("utf-16-le")) is False
