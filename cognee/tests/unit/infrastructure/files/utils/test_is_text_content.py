"""Tests for is_text_content and decode_text_content.

Verifies that UTF-16 and UTF-32 files (both with BOM and without BOM) are
recognized as text rather than binary, while binary file formats remain
properly recognized as non-text.
"""

import pytest

from cognee.infrastructure.files.utils.is_text_content import (
    decode_text_content,
    is_text_content,
)


@pytest.mark.parametrize(
    "label, content",
    [
        ("utf8_plain", b"Hello world, this is a plain UTF-8 text file.\nWith multiple lines."),
        ("utf8_bom", b"\xef\xbb\xbfHello world with UTF-8 BOM."),
        ("utf16_le_bom", "Hello world, UTF-16 LE with BOM.\n".encode("utf-16")),
        (
            "utf16_be_bom",
            b"\xfe\xff" + "Hello world, UTF-16 BE with BOM.\n".encode("utf-16-be"),
        ),
        ("utf16_le_nobom", "Hello world, UTF-16 LE without BOM.\n".encode("utf-16-le")),
        ("utf16_be_nobom", "Hello world, UTF-16 BE without BOM.\n".encode("utf-16-be")),
        ("utf32_le_bom", b"\xff\xfe\x00\x00" + "Hello world".encode("utf-32-le")),
        ("utf32_be_bom", b"\x00\x00\xfe\xff" + "Hello world".encode("utf-32-be")),
        ("empty", b""),
    ],
)
def test_is_text_content_accepts_text_encodings(label, content):
    assert is_text_content(content) is True, f"Failed for {label}"


@pytest.mark.parametrize(
    "label, content",
    [
        ("pe_exe_header", b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff"),
        ("elf_header", b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00"),
        ("png_header", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"),
        ("jpeg_header", b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"),
        ("zeroes", b"\x00" * 256),
        ("random_bytes", bytes(range(256))),
    ],
)
def test_is_text_content_rejects_binary_content(label, content):
    assert is_text_content(content) is False, f"Failed to reject binary {label}"


@pytest.mark.parametrize(
    "label, raw_bytes, expected_substring",
    [
        (
            "utf8_plain",
            b"Contact: ada@example.com or +1 555 123 4567.",
            "ada@example.com",
        ),
        (
            "utf8_bom",
            b"\xef\xbb\xbfContact: ada@example.com.",
            "ada@example.com",
        ),
        (
            "utf16_bom",
            "Contact: ada@example.com or +1 555 123 4567.".encode("utf-16"),
            "ada@example.com",
        ),
        (
            "utf16_le_nobom",
            "Contact: ada@example.com or +1 555 123 4567.".encode("utf-16-le"),
            "ada@example.com",
        ),
        (
            "utf16_be_nobom",
            "Contact: ada@example.com or +1 555 123 4567.".encode("utf-16-be"),
            "ada@example.com",
        ),
    ],
)
def test_decode_text_content(label, raw_bytes, expected_substring):
    decoded = decode_text_content(raw_bytes)
    assert expected_substring in decoded
    # Must not contain interspersed null bytes
    assert "\x00" not in decoded
