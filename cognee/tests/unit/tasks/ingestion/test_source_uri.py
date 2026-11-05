import io

from starlette.datastructures import UploadFile

from cognee.api.v1.serve.cloud_client import _text_upload_filename
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.ingest_data import _source_uri_from_input


def test_source_uri_preserves_remote_origin():
    assert (
        _source_uri_from_input("https://example.test/reports/2026?q=1")
        == "https://example.test/reports/2026?q=1"
    )


def test_source_uri_never_treats_raw_text_as_a_locator():
    assert _source_uri_from_input("This is ordinary source text, not a file.") is None


def test_source_uri_normalizes_local_file(tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("report")

    assert _source_uri_from_input(str(source)) == source.resolve().as_uri()


def test_source_uri_skips_url_lookalike_text_when_literal(tmp_path):
    # A literal_text DataItem's content is provider data, not a locator cognee
    # resolved itself, so a URL- or path-shaped body must not surface as a
    # source_uri even though the same string would otherwise qualify.
    existing_file = tmp_path / "report.txt"
    existing_file.write_text("report")

    assert (
        _source_uri_from_input(DataItem(data="https://example.test/x", literal_text=True)) is None
    )
    assert _source_uri_from_input(DataItem(data="s3://bucket/key", literal_text=True)) is None
    assert _source_uri_from_input(DataItem(data=str(existing_file), literal_text=True)) is None


def test_source_uri_still_resolves_non_literal_data_item():
    item = DataItem(data="https://example.test/x")

    assert _source_uri_from_input(item) == "https://example.test/x"


def test_source_uri_resolves_an_open_file(tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("report")

    with open(source, "rb") as file:
        assert _source_uri_from_input(file) == source.resolve().as_uri()


def test_source_uri_is_none_for_an_upload():
    # An upload's filename is the client's, not a path on the server.
    upload = UploadFile(file=io.BytesIO(b"report"), filename="report.pdf")

    assert _source_uri_from_input(upload) is None


def test_source_uri_is_none_for_text_sent_by_a_remote_client():
    # After serve(), add() uploads raw text as a file named after its content.
    text = "Raw text added through a remote client."
    upload = UploadFile(file=io.BytesIO(text.encode()), filename=_text_upload_filename(text))

    assert _source_uri_from_input(upload) is None
