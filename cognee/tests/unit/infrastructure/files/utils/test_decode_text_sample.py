import pytest

from cognee.infrastructure.files.utils.decode_text_sample import decode_text_sample
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
