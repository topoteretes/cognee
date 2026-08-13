"""CSV source creation must handle fsspec URLs, not just local paths.

``os.path.abspath("s3://bucket/x.csv")`` treats the URL as a relative local
path and mangles it into ``<cwd>/s3:/bucket/x.csv`` — which is exactly what
the old implementation did, silently pointing dlt's filesystem reader at a
nonexistent local directory. URL inputs must be split into parent-URL +
filename and passed through untouched.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from cognee.tasks.ingestion.create_dlt_source import create_dlt_source_from_csv, is_csv_path


def _capture_filesystem_call(csv_path: str) -> dict:
    """Run create_dlt_source_from_csv with dlt's readers mocked; return the
    kwargs the filesystem source was built with."""
    filesystem_mock = MagicMock()
    with (
        patch("dlt.sources.filesystem.filesystem", filesystem_mock),
        patch("dlt.sources.filesystem.read_csv", MagicMock()),
    ):
        create_dlt_source_from_csv(csv_path)
    return filesystem_mock.call_args.kwargs


@pytest.mark.parametrize(
    "csv_path, expected_bucket_url, expected_glob",
    [
        ("s3://my-bucket/data.csv", "s3://my-bucket", "data.csv"),
        ("s3://my-bucket/nested/dir/rows.csv", "s3://my-bucket/nested/dir", "rows.csv"),
        ("gs://gcs-bucket/exports/table.csv", "gs://gcs-bucket/exports", "table.csv"),
        ("file:///var/data/local.csv", "file:///var/data", "local.csv"),
        ("memory://scratch/mem.csv", "memory://scratch", "mem.csv"),
    ],
)
def test_url_inputs_pass_through_unmangled(csv_path, expected_bucket_url, expected_glob):
    kwargs = _capture_filesystem_call(csv_path)
    assert kwargs["bucket_url"] == expected_bucket_url
    assert kwargs["file_glob"] == expected_glob


def test_local_absolute_path_resolves_to_file_url():
    kwargs = _capture_filesystem_call("/tmp/some/dir/data.csv")
    assert kwargs["bucket_url"] == "file:///tmp/some/dir"
    assert kwargs["file_glob"] == "data.csv"


def test_local_relative_path_resolves_against_cwd():
    kwargs = _capture_filesystem_call("relative.csv")
    assert kwargs["bucket_url"] == f"file://{os.getcwd()}"
    assert kwargs["file_glob"] == "relative.csv"


def test_is_csv_path_accepts_urls_and_rejects_http():
    assert is_csv_path("s3://bucket/data.csv")
    assert is_csv_path("/local/data.csv")
    assert is_csv_path("file:///local/data.csv")
    assert not is_csv_path("https://example.com/data.csv")
    assert not is_csv_path("http://example.com/data.csv")
    assert not is_csv_path("s3://bucket/data.parquet")
