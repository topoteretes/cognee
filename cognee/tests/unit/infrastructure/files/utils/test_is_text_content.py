"""``is_text_content`` must trust a BOM over the null-byte heuristic.

UTF-16/UTF-32 text is riddled with null bytes, so checking for nulls before
checking for a byte order mark misjudged every BOM-marked file as binary and
presort's PII scan skipped it (topoteretes/cognee#6563).
"""

import pytest

from cognee.infrastructure.files.utils.is_text_content import is_text_content

_TEXT = "Contact: jane.doe@example.com\n"


@pytest.mark.parametrize(
    "encoding",
    ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-32"],
)
def test_bom_marked_text_is_text(encoding):
    content = _TEXT.encode(encoding)
    if encoding in ("utf-16-le", "utf-16-be"):
        content = (b"\xff\xfe" if encoding == "utf-16-le" else b"\xfe\xff") + content
    assert is_text_content(content)


def test_binary_with_null_bytes_is_not_text():
    assert not is_text_content(b"\x89PNG\r\n\x1a\n" + bytes(range(256)))


def test_plain_utf8_is_text():
    assert is_text_content(_TEXT.encode("utf-8"))
