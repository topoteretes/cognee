"""BOM-marked UTF-16/UTF-32 .txt files must ingest instead of raising.

Regression for #6839: extract_text_from_file decoded .txt as strict UTF-8,
so a BOM-marked UTF-16/UTF-32 file raised UnicodeDecodeError at ingest even
though presort (#6641) already reads the same file as text.
"""

import io

import pytest
from filetype.types.base import Type

from cognee.infrastructure.files.utils.extract_text_from_file import extract_text_from_file


def _txt_type() -> Type:
    return Type(mime="text/plain", extension="txt")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("Hello cognee".encode("utf-8"), "Hello cognee"),
        (b"\xef\xbb\xbf" + "Hello cognee".encode("utf-8"), "Hello cognee"),
        ("Hello cognee".encode("utf-16"), "Hello cognee"),
        (b"\xff\xfe" + "Hello cognee".encode("utf-16-le"), "Hello cognee"),
        (b"\xfe\xff" + "Hello cognee".encode("utf-16-be"), "Hello cognee"),
        ("Hello cognee".encode("utf-32"), "Hello cognee"),
        (b"\xff\xfe\x00\x00" + "Hello cognee".encode("utf-32-le"), "Hello cognee"),
        (b"\x00\x00\xfe\xff" + "Hello cognee".encode("utf-32-be"), "Hello cognee"),
    ],
)
def test_txt_bom_variants(payload: bytes, expected: str):
    assert extract_text_from_file(io.BytesIO(payload), _txt_type()) == expected


def test_txt_invalid_utf8_still_raises():
    with pytest.raises(UnicodeDecodeError):
        extract_text_from_file(io.BytesIO(b"not utf-8 \xff"), _txt_type())
