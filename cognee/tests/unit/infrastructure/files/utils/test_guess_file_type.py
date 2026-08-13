"""``guess_file_type`` must detect extensions case-insensitively.

Extensions are case-insensitive by convention, but the extension-based mapping
only matched lowercase. For types with no magic-number signature
(csv/md/json/xml/yaml), an uppercase extension fell through to
``filetype.guess``, returned ``None``, and was mislabeled ``text/plain`` — so a
``data.CSV`` was classified as a plain ``TextDocument`` instead of a
``CsvDocument`` (the crash for these was fixed in #3662, but the wrong type
survived). PDFs and other magic-number formats are unaffected either way.
"""

import io

import pytest

from cognee.infrastructure.files.utils.guess_file_type import guess_file_type

# Content with no magic number, so detection depends on the extension.
_CSV_BYTES = b"name,age\nJohn,30\nJane,25\n"


@pytest.mark.parametrize(
    "name, expected_mime, expected_ext",
    [
        ("data.csv", "text/csv", "csv"),
        ("data.CSV", "text/csv", "csv"),
        ("notes.md", "text/markdown", "md"),
        ("notes.MD", "text/markdown", "md"),
        ("conf.json", "application/json", "json"),
        ("conf.JSON", "application/json", "json"),
        ("doc.XML", "application/xml", "xml"),
        ("conf.YAML", "application/yaml", "yaml"),
    ],
)
def test_extension_detection_is_case_insensitive(name, expected_mime, expected_ext):
    file_type = guess_file_type(io.BytesIO(_CSV_BYTES), name=name)
    assert file_type.mime == expected_mime
    assert file_type.extension == expected_ext


def test_uppercase_txt_is_still_plain_text():
    file_type = guess_file_type(io.BytesIO(b"just some text"), name="README.TXT")
    assert file_type.mime == "text/plain"
