"""NUL in a Drive document on real PostgreSQL: the load it breaks and what cures it.

PostgreSQL rejects NUL in text, so one document with it fails the load of its whole dlt
package. ``strip_nul_from_rows`` is the step ``ingest_dlt_source`` adds to every document
source, and these tests run it the way it runs there. Runs against the test database of the
sibling tests (default: cognee:cognee@localhost:5432/cognee_db, overridable with DB_HOST,
DB_PORT, DB_USERNAME, DB_PASSWORD and DB_NAME) and skips if it is unreachable. Each test owns
two unique schemas and a temporary pipeline. The Drive client and the connector's listing and
cursor lookups are stubbed; decoding, row construction and DLT loads are real.
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
from sqlalchemy import URL, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.schema import DropSchema

from cognee.tasks.ingestion.connectors import google_drive as drive
from cognee.tasks.ingestion.connectors import obsidian_source
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_COLUMN_HINTS
from cognee.tasks.ingestion.ingest_dlt_source import strip_nul_from_rows


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    credentials = {
        "drivername": "postgresql",
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", "5432")),
        "username": os.environ.get("DB_USERNAME", "cognee"),
        "password": os.environ.get("DB_PASSWORD", "cognee"),
        "database": os.environ.get("DB_NAME", "cognee_db"),
    }
    engine = create_engine(URL.create(**credentials))
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError:
        engine.dispose()
        pytest.skip("Postgres not reachable")
    engine.dispose()
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


def _run_unstripped(pipeline, source):
    # How a sync staged before the step existed, and how any source that skips it would.
    return pipeline.run(
        source, write_disposition="merge", primary_key="id", columns=DOCUMENT_COLUMN_HINTS
    )


def _run(pipeline, source):
    strip_nul_from_rows(source)
    return _run_unstripped(pipeline, source)


def _rows(pipeline):
    with pipeline.sql_client() as sql:
        table = sql.make_qualified_table_name("folder_rows")
        return sql.execute_sql(f"SELECT id, title, content FROM {table} ORDER BY id")


def _state(pipeline):
    return pipeline.state["sources"][pipeline.default_schema_name]["resources"]["folder_rows"]


def test_one_nul_document_fails_the_load_of_its_neighbours(pipeline, remote):
    with pytest.raises(PipelineStepFailed, match="NUL"):
        _run_unstripped(pipeline, remote.source())

    assert len(pipeline.list_normalized_load_packages()) == 1
    assert _rows(pipeline) == []  # not even the documents without NUL were loaded


def test_another_connector_with_a_nul_note_fails_and_loads_with_the_same_step(pipeline, tmp_path):
    # The rule is not Drive's: an Obsidian note with a stray NUL byte blocked its vault the same way.
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Note.md").write_bytes(b"alpha\x00beta\n")
    (vault / "Other.md").write_bytes(b"clean note\n")

    with pytest.raises(PipelineStepFailed, match="NUL"):
        _run_unstripped(pipeline, obsidian_source(vault_path=vault, resource_name="notes"))
    pipeline.drop_pending_packages()

    _run(pipeline, obsidian_source(vault_path=vault, resource_name="notes"))
    with pipeline.sql_client() as sql:
        table = sql.make_qualified_table_name("notes")
        rows = sql.execute_sql(f"SELECT id, content FROM {table} ORDER BY id")
    assert rows == [("obsidian:Note.md", "alphabeta\n"), ("obsidian:Other.md", "clean note\n")]


@pytest.mark.parametrize("content", ["pre\x00posle\nć 😀\t", "\x00"])
def test_mixed_documents_load_and_unchanged_sync_preserves_rows(pipeline, remote, content):
    remote.content["nul"] = content
    source = remote.source()
    _run(pipeline, source)

    expected = [
        ("after", "after", "ispravan posle"),
        ("before", "before", "ispravan pre"),
        ("nul", "Naslovdokumenta", content.replace("\x00", "")),
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
    # A package extracted before the step existed, which the load then refuses.
    with pytest.raises(PipelineStepFailed, match="NUL"):
        _run_unstripped(pipeline, remote.source())
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
        ("nul", "Naslovdokumenta", "preposle\nć 😀\t"),
    ]
    assert _state(pipeline)["page_token"] == "t1"
    assert not pipeline.list_normalized_load_packages()
