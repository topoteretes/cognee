"""Remote CSVs are localized through cognee's storage layer for dlt staging.

The old implementation ran every CSV path through os.path.abspath, which
mangles URLs ("s3://bucket/x.csv" -> "<cwd>/s3:/bucket/x.csv") and silently
pointed dlt's filesystem reader at a nonexistent local directory. The fix
delegates the local-vs-S3 distinction to cognee's file layer: s3:// CSVs are
downloaded via open_data_file (S3 credentials come from cognee's S3Config /
IAM chain, never a parallel fsspec configuration) into a temp copy scoped to
staging ingestion, and dlt only ever reads local files.
"""

import io
import os
import pathlib
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The unit CI matrices install the base extras only — dlt is optional there,
# and these tests patch dlt.sources.filesystem / drive dlt-gated resolution.
pytest.importorskip("dlt")

from cognee.tasks.ingestion.create_dlt_source import (
    create_dlt_source_from_csv,
    download_csv_for_staging,
    is_csv_path,
    is_remote_csv_path,
)


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


def test_local_absolute_path_resolves_to_file_url():
    kwargs = _capture_filesystem_call("/tmp/some/dir/data.csv")
    assert kwargs["bucket_url"] == "file:///tmp/some/dir"
    assert kwargs["file_glob"] == "data.csv"


def test_local_relative_path_resolves_against_cwd():
    kwargs = _capture_filesystem_call("relative.csv")
    assert kwargs["bucket_url"] == f"file://{os.getcwd()}"
    assert kwargs["file_glob"] == "relative.csv"


def test_file_url_is_stripped_to_local_path():
    kwargs = _capture_filesystem_call("file:///var/data/local.csv")
    assert kwargs["bucket_url"] == "file:///var/data"
    assert kwargs["file_glob"] == "local.csv"


def test_csv_path_predicates():
    assert is_csv_path("s3://bucket/data.csv")
    assert is_csv_path("/local/data.csv")
    assert is_csv_path("file:///local/data.csv")
    assert not is_csv_path("https://example.com/data.csv")
    assert not is_csv_path("http://example.com/data.csv")

    assert is_remote_csv_path("s3://bucket/data.csv")
    assert not is_remote_csv_path("/local/data.csv")
    assert not is_remote_csv_path("file:///local/data.csv")
    assert not is_remote_csv_path("s3://bucket/data.parquet")


@pytest.mark.asyncio
async def test_download_csv_for_staging_uses_cognee_file_layer(tmp_path, monkeypatch):
    """The download must go through open_data_file — cognee's canonical
    local-vs-S3 entry — and land in a per-download subdirectory."""
    opened_urls = []

    @asynccontextmanager
    async def fake_open_data_file(file_path, mode="rb", **kwargs):
        opened_urls.append((file_path, mode))
        yield io.BytesIO(b"id,name\n1,anemometer\n")

    open_data_file_module = sys.modules["cognee.infrastructure.files.utils.open_data_file"]
    monkeypatch.setattr(open_data_file_module, "open_data_file", fake_open_data_file)

    local_path = await download_csv_for_staging("s3://bucket/exports/rows.csv", str(tmp_path))

    assert opened_urls == [("s3://bucket/exports/rows.csv", "rb")]
    assert pathlib.Path(local_path).read_bytes() == b"id,name\n1,anemometer\n"
    assert pathlib.Path(local_path).name == "rows.csv"
    assert pathlib.Path(local_path).parent.parent == tmp_path


@pytest.mark.asyncio
async def test_resolve_localizes_remote_csv_and_cleans_up(monkeypatch):
    """resolve_dlt_sources must download s3 CSVs before source creation and
    remove the temp copy once staging consumption is over."""
    # The package __init__ re-exports resolve_dlt_sources (the function),
    # shadowing the submodule on plain import — go through sys.modules.
    import cognee.tasks.ingestion.resolve_dlt_sources  # noqa: F401

    resolve_module = sys.modules["cognee.tasks.ingestion.resolve_dlt_sources"]

    seen = {}

    async def fake_download(csv_url, temp_dir):
        seen["url"] = csv_url
        seen["temp_dir"] = temp_dir
        return os.path.join(temp_dir, "localized.csv")

    def fake_create(local_path):
        seen["created_from"] = local_path
        return MagicMock(name="not_a_dlt_object")

    monkeypatch.setattr(resolve_module, "download_csv_for_staging", fake_download)
    monkeypatch.setattr(resolve_module, "create_dlt_source_from_csv", fake_create)

    data = ["s3://bucket/exports/rows.csv"]
    result, cleanup = await resolve_module.resolve_dlt_sources(
        data, dataset_name="ds", user=MagicMock()
    )

    assert seen["url"] == "s3://bucket/exports/rows.csv"
    assert seen["created_from"] == os.path.join(seen["temp_dir"], "localized.csv")
    # The mocked source is not a real dlt object, so resolution falls through
    # unchanged — and the remote-CSV temp directory must already be gone.
    assert result == data
    assert cleanup is None
    assert not os.path.exists(seen["temp_dir"])
