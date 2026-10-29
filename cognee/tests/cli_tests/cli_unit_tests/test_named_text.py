"""``--name`` names the one inline text of a ``cognee add`` / ``cognee remember`` call."""

import pytest

from cognee.cli.named_text import check_name, name_inline_text, upload_filename
from cognee.tasks.ingestion.data_item import DataItem


def test_one_inline_text_may_be_named(tmp_path):
    upload = tmp_path / "report.pdf"
    upload.write_bytes(b"%PDF")

    check_name(["some words", str(upload)], "Q3 plan")  # one text, one file: fine


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (["a", "b"], "exactly one inline text item; got 2"),
        ([], "got 0"),
        # A URL is fetched by cognee, not stored as text: nothing to name.
        (["https://example.com"], "got 0"),
        (["s3://bucket/key"], "got 0"),
    ],
)
def test_a_name_is_refused_when_it_could_mean_several_things(items, message):
    with pytest.raises(ValueError, match=message):
        check_name(items, "Q3 plan")


def test_an_empty_name_is_refused():
    with pytest.raises(ValueError, match="must not be empty"):
        check_name(["a"], "   ")


def test_no_name_is_always_fine():
    check_name(["a", "b", "c"], None)


def test_a_url_or_directory_next_to_the_text_does_not_count(tmp_path):
    check_name(["some words", "https://example.com", str(tmp_path)], "Q3 plan")
    assert name_inline_text(["https://example.com", "some words"], "Q3 plan")[0] == (
        "https://example.com"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Q3 plan", "Q3 plan.txt"),
        ("notes.txt", "notes.txt"),
        ("roadmap.docx", "roadmap.docx.txt"),
        ("../../etc/passwd", "passwd.txt"),
        ("dir\\sub\\note", "note.txt"),
        # The same sanitising as the in-process lane (text_file_name): control
        # characters and leading dots go, and the name fits a 255-byte basename.
        ("\x00.hidden\x1fname", "hiddenname.txt"),
        ("日本語" * 90, "日本語" * 21 + "日本.txt"),
    ],
)
def test_upload_filename_is_a_txt_basename(name, expected):
    assert upload_filename(name) == expected


def test_named_text_becomes_a_data_item_and_the_rest_passes_through(tmp_path):
    upload = tmp_path / "report.pdf"
    upload.write_bytes(b"%PDF")

    items = name_inline_text([str(upload), "some words"], " Q3 plan ")

    assert items[0] == str(upload)
    assert isinstance(items[1], DataItem)
    assert items[1].data == "some words"
    assert items[1].name == "Q3 plan"


def test_without_a_name_nothing_is_wrapped():
    assert name_inline_text(["a", "b"], None) == ["a", "b"]
