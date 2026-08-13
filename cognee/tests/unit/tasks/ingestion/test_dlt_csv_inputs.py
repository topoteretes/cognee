"""CSV inputs to the DLT auto-detection: paths, uploads, and s3 locations.

resolve_dlt_sources auto-routes CSVs to the DLT manifest path. Three input
shapes must all land there with a filename-derived source name (the manifest
identity is seeded from it):

1. local path strings (and file:// URIs) — read in place, nothing spooled
2. file-like uploads (API UploadFile / SDK binary handles) — spooled to a
   temp dir dlt's filesystem source can read
3. s3:// paths — downloaded through cognee's storage layer, then spooled

Naming per file (instead of dlt's fixed "_read_csv" pipe name) is what keeps
two CSVs in one dataset from collapsing into a single manifest identity.
"""

import io
import logging
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import cognee.infrastructure.files.utils.open_data_file as open_data_file_module
from cognee.tasks.ingestion.create_dlt_source import (
    create_dlt_source_from_csv,
    csv_source_name,
    is_csv_upload,
)
from cognee.tasks.ingestion.resolve_dlt_sources import (
    _log_structured_inputs_without_dlt,
    _normalize_structured_inputs,
)

CSV_BYTES = b"id,name\n1,Ada\n2,Alan\n"


def _upload(filename="people.csv", content=CSV_BYTES):
    """Duck-typed API upload: .file + .filename, like starlette's UploadFile."""
    return SimpleNamespace(file=io.BytesIO(content), filename=filename)


def _csv_file(tmp_path, name="people.csv", content=CSV_BYTES):
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


class TestCsvSourceName:
    def test_stem_is_sanitized_and_lowercased(self):
        assert csv_source_name("My Data-2024.csv") == "my_data_2024"

    def test_path_is_reduced_to_basename(self):
        assert csv_source_name("/some/dir/people.csv") == "people"

    def test_unusable_stem_falls_back(self):
        assert csv_source_name("---.csv") == "csv_source"


class TestIsCsvUpload:
    def test_api_upload_shape(self):
        assert is_csv_upload(_upload())

    def test_binary_handle_shape(self, tmp_path):
        with open(_csv_file(tmp_path), "rb") as handle:
            assert is_csv_upload(handle)

    def test_strings_are_not_uploads(self):
        assert not is_csv_upload("people.csv")

    def test_non_csv_filename_rejected(self):
        assert not is_csv_upload(SimpleNamespace(file=io.BytesIO(), filename="notes.txt"))

    def test_filename_less_object_rejected(self):
        assert not is_csv_upload(io.BytesIO(CSV_BYTES))


class TestCreateDltSourceFromCsv:
    def test_resource_named_after_file(self, tmp_path):
        source = create_dlt_source_from_csv(_csv_file(tmp_path, "Employee Roster.csv"))
        assert source.name == "employee_roster"


@pytest.mark.asyncio
class TestNormalizeStructuredInputs:
    async def test_local_path_passes_through_without_spooling(self, tmp_path):
        items, spool_dir = await _normalize_structured_inputs([_csv_file(tmp_path)], None)
        assert spool_dir is None
        assert items[0].name == "people"

    async def test_file_uri_is_unwrapped(self, tmp_path):
        items, spool_dir = await _normalize_structured_inputs(
            [f"file://{_csv_file(tmp_path)}"], None
        )
        assert spool_dir is None
        assert items[0].name == "people"

    async def test_upload_is_spooled_and_named_after_filename(self, tmp_path):
        items, spool_dir = await _normalize_structured_inputs(
            [_upload("Employee Roster.csv")], None
        )
        try:
            assert spool_dir is not None
            assert items[0].name == "employee_roster"
            spooled = os.path.join(spool_dir, "employee_roster.csv")
            with open(spooled, "rb") as spooled_file:
                assert spooled_file.read() == CSV_BYTES
        finally:
            if spool_dir:
                import shutil

                shutil.rmtree(spool_dir, ignore_errors=True)

    async def test_non_structured_items_pass_through(self):
        items, spool_dir = await _normalize_structured_inputs(["a plain note"], None)
        assert spool_dir is None
        assert items == ["a plain note"]

    async def test_same_name_uploads_fail_loudly(self):
        with pytest.raises(ValueError, match="same source name"):
            await _normalize_structured_inputs([_upload("dup.csv"), _upload("dup.csv")], None)

    async def test_s3_path_is_downloaded_via_storage_layer(self, monkeypatch):
        opened = []

        @asynccontextmanager
        async def fake_open_data_file(file_path, mode="rb", **kwargs):
            opened.append((file_path, mode))
            yield io.BytesIO(CSV_BYTES)

        monkeypatch.setattr(open_data_file_module, "open_data_file", fake_open_data_file)

        items, spool_dir = await _normalize_structured_inputs(
            ["s3://bucket/exports/orders.csv"], None
        )
        try:
            assert opened == [("s3://bucket/exports/orders.csv", "rb")]
            assert items[0].name == "orders"
            spooled = os.path.join(spool_dir, "orders.csv")
            with open(spooled, "rb") as spooled_file:
                assert spooled_file.read() == CSV_BYTES
        finally:
            if spool_dir:
                import shutil

                shutil.rmtree(spool_dir, ignore_errors=True)


class TestDltMissingWarning:
    def test_warns_for_structured_inputs(self, caplog):
        with caplog.at_level(logging.WARNING):
            _log_structured_inputs_without_dlt(
                [
                    "data/people.csv",
                    _upload("roster.csv"),
                    "postgresql://user:secret@host/db",
                    "a plain note",
                ]
            )
        assert "3 structured input(s)" in caplog.text
        assert "people.csv" in caplog.text
        assert "roster.csv" in caplog.text
        # Connection strings can embed credentials — never logged verbatim.
        assert "secret" not in caplog.text
        assert "<connection string>" in caplog.text

    def test_silent_when_nothing_structured(self, caplog):
        with caplog.at_level(logging.WARNING):
            _log_structured_inputs_without_dlt(["a plain note", 42])
        assert caplog.text == ""
