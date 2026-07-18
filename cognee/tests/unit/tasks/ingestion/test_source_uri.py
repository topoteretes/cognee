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
