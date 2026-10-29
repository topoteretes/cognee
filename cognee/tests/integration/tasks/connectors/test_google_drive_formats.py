"""Real Office parsers through Drive and DLT, including a pre-support checkpoint.

No Google or LLM credentials: only the remote listing/downloads are replaced.
The docling-slim CI job runs this file with the actual Office dependencies.
"""

import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import dlt
import pytest
from dlt.pipeline.exceptions import PipelineStepFailed

from cognee.tasks.ingestion.connectors import google_drive as drive
from cognee.tasks.ingestion.ingest_dlt_source import _row_content_hash


@pytest.fixture(params=["docx", "xlsx", "pptx", "slides"])
def format_case(request, monkeypatch):
    extension = request.param
    # CI installs the docling extra, so there a missing module must fail, not skip.
    if extension != "slides" and not os.getenv("CI"):
        pytest.importorskip("docling", reason="real Office conversion needs the docling extra")
    fixtures = Path(__file__).resolve().parents[3] / "test_data"
    formats = {
        "docx": (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ("Paragraph",),
        ),
        "xlsx": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ("| a", "2. e"),
        ),
        "pptx": (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ("A Slide with a Table", "Tangerine"),
        ),
        "slides": (drive.GOOGLE_SLIDES_MIME_TYPE, ("First slide", "Second slide")),
    }
    mime, expected = formats[extension]
    files = [
        {"id": "existing", "name": "Existing", "mimeType": drive.GOOGLE_DOC_MIME_TYPE},
        {"id": "new_format", "name": f"Example.{extension}", "mimeType": mime},
    ]
    for file in files:
        file.update(modifiedTime="v1", webViewLink=f"https://drive/{file['id']}")
    old_row = {
        "id": "existing",
        "title": "Existing",
        "content": "Already ingested document",
        "url": "https://drive/existing",
        "_deleted": False,
    }
    downloads = []

    class Service:
        def files(self):
            return self

        def export(self, fileId, mimeType):
            assert mimeType == "text/plain"
            downloads.append(fileId)
            data = (
                old_row["content"].encode()
                if fileId == "existing"
                else b"First slide\nSecond slide"
            )
            return SimpleNamespace(execute=lambda **kwargs: data)

        def get_media(self, fileId, supportsAllDrives):
            assert supportsAllDrives and fileId == "new_format"
            downloads.append(fileId)
            return SimpleNamespace(execute=lambda: (fixtures / f"example.{extension}").read_bytes())

    listing = Mock(side_effect=lambda *args: iter(files))
    monkeypatch.setattr(drive, "_list_files_in_scope", listing)
    monkeypatch.setattr(drive, "_get_start_page_token", lambda *args: "t1")
    monkeypatch.setattr(drive, "_list_changed_file_ids", lambda *args: (set(), set(), "t2"))
    monkeypatch.setattr(
        drive, "_stream_media", lambda request, buffer: buffer.write(request.execute())
    )
    return SimpleNamespace(
        files=files,
        old_row=old_row,
        expected=expected,
        downloads=downloads,
        listing=listing,
        source=lambda: drive.google_drive_source(
            folder_id="folder", resource_name="folder_files", service=Service()
        ),
    )


def _state(pipeline):
    return pipeline.state["sources"][pipeline.default_schema_name]["resources"]["folder_files"]


def _rows(pipeline):
    with pipeline.sql_client() as sql:
        rows = sql.execute_sql(
            "SELECT id, title, content, url, _deleted FROM folder_files ORDER BY id"
        )
    return [dict(zip(("id", "title", "content", "url", "_deleted"), row)) for row in rows]


@pytest.mark.parametrize("interrupted", [False, True])
def test_old_checkpoint_backfills_real_formats_without_duplicate_documents(
    tmp_path, monkeypatch, format_case, interrupted
):
    case = format_case
    pipeline = dlt.pipeline(
        pipeline_name="drive_formats",
        dataset_name="drive",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'drive.db'}"),
        pipelines_dir=str(tmp_path / "pipelines"),
    )

    @dlt.resource(name="folder_files", primary_key="id", write_disposition="merge")
    def legacy():
        dlt.current.resource_state().update(known_ids=["existing", "new_format"], page_token="t0")
        yield case.old_row

    pipeline.run(legacy())
    old_rows = _rows(pipeline)
    old_state = deepcopy(_state(pipeline))
    if interrupted:

        def broken_listing(*args):
            yield case.files[0]
            raise RuntimeError("listing interrupted")

        with monkeypatch.context() as patch:
            patch.setattr(drive, "_list_files_in_scope", broken_listing)
            with pytest.raises(PipelineStepFailed, match="listing interrupted"):
                pipeline.run(case.source())
        assert _state(pipeline) == old_state
        assert _rows(pipeline) == old_rows
        case.downloads.clear()

    source = case.source()
    pipeline.run(source)
    rows = _rows(pipeline)
    assert [row["id"] for row in rows] == ["existing", "new_format"]
    assert _row_content_hash(rows[0]) == _row_content_hash(old_rows[0])
    assert all(marker in rows[1]["content"] for marker in case.expected)
    assert case.downloads == ["existing", "new_format"]
    assert source.cognee_sync_stats["failed"] == source.cognee_sync_stats["skipped"] == 0
    assert _state(pipeline)["format_version"] == drive.FORMAT_SUPPORT_VERSION
    assert _state(pipeline)["page_token"] == "t1"
    assert not pipeline.list_normalized_load_packages()

    # A fresh pipeline object must retain the migration checkpoint and not re-parse.
    pipeline = dlt.attach("drive_formats", pipelines_dir=str(tmp_path / "pipelines"))
    case.downloads.clear()
    case.listing.reset_mock()
    pipeline.run(
        case.source(),
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'drive.db'}"),
    )
    assert case.downloads == []
    case.listing.assert_not_called()
    assert _rows(pipeline) == rows
    assert _state(pipeline)["page_token"] == "t2"
