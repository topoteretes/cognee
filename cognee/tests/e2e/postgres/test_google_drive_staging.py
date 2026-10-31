"""Drive text normalization and pending-package recovery on real PostgreSQL.

Needs DB_HOST/PORT/USERNAME/PASSWORD/NAME pointing at a test database and fails
without them. Each test owns two unique schemas and a temporary pipeline. The Drive
client and the connector's listing and cursor lookups are stubbed; decoding, row
construction and DLT loads are real.
"""

import os
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import dlt
import pytest
from dlt.common.configuration.container import Container
from dlt.common.pipeline import PipelineContext
from dlt.pipeline.exceptions import PipelineStepFailed
from sqlalchemy import URL, create_engine
from sqlalchemy.schema import DropSchema

from cognee.tasks.ingestion.connectors import google_drive as drive
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_COLUMN_HINTS


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    credentials = {
        "drivername": "postgresql",
        "host": os.environ["DB_HOST"],
        "port": int(os.environ["DB_PORT"]),
        "username": os.environ["DB_USERNAME"],
        "password": os.environ["DB_PASSWORD"],
        "database": os.environ["DB_NAME"],
    }
    schema = f"drive_{uuid4().hex}"
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / "dlt"))
    Container()[PipelineContext].deactivate()
    instance = dlt.pipeline(
        pipeline_name="drive_nul",
        dataset_name=schema,
        destination=dlt.destinations.sqlalchemy(credentials=credentials),
        pipelines_dir=str(tmp_path / "pipelines"),
    )
    try:
        yield instance
    finally:
        Container()[PipelineContext].deactivate()
        engine = create_engine(URL.create(**credentials))
        try:
            with engine.begin() as connection:
                for name in (schema, f"{schema}_staging"):
                    connection.execute(DropSchema(name, cascade=True, if_exists=True))
        finally:
            engine.dispose()


@pytest.fixture
def remote(monkeypatch):
    files = {
        file_id: {
            "id": file_id,
            "name": file_id,
            "mimeType": drive.GOOGLE_DOC_MIME_TYPE,
            "parents": ["folder"],
            "modifiedTime": "v1",
            "webViewLink": f"https://drive/{file_id}",
        }
        for file_id in ("before", "nul", "after")
    }
    files["nul"]["name"] = "Naslov\x00dokumenta"
    remote = SimpleNamespace(
        files=files,
        content={
            "before": "ispravan pre",
            "nul": "pre\x00posle\nć 😀\t",
            "after": "ispravan posle",
        },
        listed=list(files.values()),
        changed=set(),
        cursor="t0",
        downloads=[],
    )

    class Service:
        def files(self):
            return self

        def get(self, fileId, **kwargs):
            return SimpleNamespace(execute=lambda **kwargs: files[fileId])

        def export(self, fileId, **kwargs):
            remote.downloads.append(fileId)
            return SimpleNamespace(execute=lambda **kwargs: remote.content[fileId].encode("utf-8"))

    service = Service()
    remote.source = lambda: drive.google_drive_source(
        folder_id="folder", resource_name="folder_rows", service=service
    )
    monkeypatch.setattr(drive, "_list_files_in_scope", lambda *args: iter(remote.listed))
    monkeypatch.setattr(drive, "_get_start_page_token", lambda *args: remote.cursor)
    monkeypatch.setattr(drive, "_get_scope_folder_ids", lambda *args: {"folder"})
    monkeypatch.setattr(drive, "_resolve_folder_id", lambda *args: "folder")
    monkeypatch.setattr(
        drive, "_list_changed_file_ids", lambda *args: (remote.changed, set(), remote.cursor)
    )
    return remote


def _run(pipeline, source):
    return pipeline.run(
        source, write_disposition="merge", primary_key="id", columns=DOCUMENT_COLUMN_HINTS
    )


def _rows(pipeline):
    with pipeline.sql_client() as sql:
        table = sql.make_qualified_table_name("folder_rows")
        return sql.execute_sql(f"SELECT id, title, content FROM {table} ORDER BY id")


def _state(pipeline):
    return pipeline.state["sources"][pipeline.default_schema_name]["resources"]["folder_rows"]


@pytest.mark.parametrize("content", ["pre\x00posle\nć 😀\t", "\x00"])
def test_mixed_documents_load_and_unchanged_sync_preserves_rows(pipeline, remote, content):
    remote.content["nul"] = content
    source = remote.source()
    _run(pipeline, source)

    expected = [
        ("after", "after", "ispravan posle"),
        ("before", "before", "ispravan pre"),
        ("nul", "Naslov\ufffddokumenta", content.replace("\x00", "\ufffd")),
    ]
    assert _rows(pipeline) == expected
    assert source.cognee_sync_stats["failed"] == source.cognee_sync_stats["skipped"] == 0
    assert _state(pipeline)["versions"] == dict.fromkeys(remote.files, "v1")
    assert _state(pipeline)["page_token"] == "t0"
    assert not pipeline.list_normalized_load_packages()

    remote.downloads.clear()
    remote.cursor = "t1"
    _run(pipeline, remote.source())
    assert remote.downloads == []
    assert _rows(pipeline) == expected
    assert _state(pipeline)["page_token"] == "t1"


@pytest.mark.skipif(
    not hasattr(dlt.Pipeline, "abort_packages"), reason="abort_packages needs dlt 1.30 or newer"
)
def test_old_nul_package_requires_abort_and_state_restore_before_resync(pipeline, remote):
    # A successful sync must survive recovery of a later, poisoned package.
    remote.listed = [remote.files["before"]]
    _run(pipeline, remote.source())
    committed_rows = _rows(pipeline)
    committed_state = deepcopy(_state(pipeline))

    remote.changed = {"nul", "after"}
    remote.cursor = "t1"
    old_source = remote.source()

    def old_unsanitized_row(row):
        # Reproduce a package extracted before the fix, without changing production code.
        if row["id"] == "nul":
            row["title"] = remote.files["nul"]["name"]
            row["content"] = remote.content["nul"]
        return row

    old_source.add_map(old_unsanitized_row)
    with pytest.raises(PipelineStepFailed, match="NUL"):
        _run(pipeline, old_source)
    pending = pipeline.list_normalized_load_packages()
    assert len(pending) == 1
    assert _rows(pipeline) == committed_rows
    assert _state(pipeline)["page_token"] == "t1"

    remote.downloads.clear()
    with pytest.raises(PipelineStepFailed, match="NUL"):
        _run(pipeline, remote.source())
    assert remote.downloads == []  # DLT retries the package before extracting the new source
    assert pipeline.list_normalized_load_packages() == pending

    # Documented operator action, using DLT 1.30's state-restoring abort.
    plan = pipeline.abort_packages(dry_run=True)
    assert plan["restore_from_load_id"] == pending[0]
    pipeline.abort_packages()
    assert not pipeline.list_normalized_load_packages()
    assert _state(pipeline) == committed_state
    assert _rows(pipeline) == committed_rows

    _run(pipeline, remote.source())
    assert sorted(remote.downloads) == ["after", "nul"]
    assert _rows(pipeline) == [
        ("after", "after", "ispravan posle"),
        ("before", "before", "ispravan pre"),
        ("nul", "Naslov\ufffddokumenta", "pre\ufffdposle\nć 😀\t"),
    ]
    assert _state(pipeline)["page_token"] == "t1"
    assert not pipeline.list_normalized_load_packages()
