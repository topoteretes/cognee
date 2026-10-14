"""``text_file_name`` turns a document title into the ``.txt`` basename text is stored under."""

import pytest

from cognee.modules.ingestion.data_types.TextData import (
    _MAX_TEXT_NAME_BYTES,
    TextData,
    text_file_name,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Q3 Plan", "Q3 Plan.txt"),
        ("notes.txt", "notes.txt"),
        ("NOTES.TXT", "NOTES.txt"),
        ("roadmap.docx", "roadmap.docx.txt"),
        ("Re: standup", "Re: standup.txt"),
        ("dir/sub/file", "file.txt"),
        ("C:\\dir\\file", "file.txt"),
        ("../../etc/passwd", "passwd.txt"),
        ("  .hidden  ", "hidden.txt"),
        ("tab\there\x00", "tabhere.txt"),
        ("trailing dots...", "trailing dots.txt"),
    ],
)
def test_names_become_safe_txt_basenames(name, expected):
    assert text_file_name(name) == expected


@pytest.mark.parametrize("name", [None, "", "   ", "...", "/", "a/b/"])
def test_unusable_names_are_none(name):
    assert text_file_name(name) is None


@pytest.mark.parametrize("title", ["x" * 1000, "日本語" * 70, "🙂" * 100])
def test_long_names_are_capped_by_bytes_to_a_storable_basename(title):
    # The filesystem limit is in bytes; a CJK or emoji title is 3-4 bytes per char.
    name = text_file_name(title)

    assert name is not None
    assert len(name.encode("utf-8")) <= _MAX_TEXT_NAME_BYTES
    assert name.endswith(".txt")
    name[: -len(".txt")].encode("utf-8").decode("utf-8")  # no split code point


def test_text_data_metadata_uses_the_name_and_keeps_the_hash():
    named = TextData("body", name="Q3 Plan").get_metadata()
    nameless = TextData("body").get_metadata()

    assert named["name"] == "Q3 Plan.txt"
    assert nameless["name"] == "text_" + nameless["content_hash"] + ".txt"
    assert named["content_hash"] == nameless["content_hash"]
    assert named["extension"] == nameless["extension"] == "txt"
