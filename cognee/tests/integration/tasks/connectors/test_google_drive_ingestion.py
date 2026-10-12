# Adapted from topoteretes/cognee-community (Apache-2.0), commit 21d0b36.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""End-to-end test: Google Drive connector through the real cognee.add()
pipeline, with the Drive API mocked (no live credentials).

Verifies the behaviors the issue asks for at the add()/Data-record layer
(no cognify(), so no LLM calls are needed):
  - initial sync creates one Data record per in-scope file
  - content-bearing rows are routed through document mode (source="google_drive"),
    so cognify would chunk + LLM-extract them rather than schema-wrap them
  - an incremental re-sync only re-processes new/changed files — an unchanged
    file's data_id is stable, so it isn't recreated
  - a file removed from Drive is forgotten (foreground orphan_cleanup fires)
"""

import pytest
import pytest_asyncio

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import google_drive as gd_source
from cognee.tasks.ingestion.connectors import google_drive_source

DATASET_NAME = "gdrive_integration_test"
DOC_MIME = "application/vnd.google-apps.document"


class _FakeRequest:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def execute(self):
        if self._error is not None:
            raise self._error
        return self._result


class _FakeHttpError(Exception):
    class _Resp:
        status = 404

    resp = _Resp()


class _FakeFilesResource:
    def __init__(self, service):
        self._service = service

    def list(self, q, fields, pageSize, pageToken=None, **kwargs):
        import re

        folder_id = re.search(r"'([^']+)' in parents", q).group(1)
        if "mimeType=" in q:
            return _FakeRequest({"files": []})  # no subfolders in this test
        files = [m for m in self._service.files_by_folder.get(folder_id, []) if not m["trashed"]]
        return _FakeRequest({"files": files})

    def get(self, fileId, fields, supportsAllDrives):
        meta = self._service.file_by_id.get(fileId)
        if meta is None:
            return _FakeRequest(error=_FakeHttpError())
        return _FakeRequest(meta)


class _FakeChangesResource:
    def __init__(self, service):
        self._service = service

    def getStartPageToken(self, supportsAllDrives, driveId=None):
        return _FakeRequest({"startPageToken": self._service.start_token})

    def list(self, pageToken, fields, **kwargs):
        return _FakeRequest(self._service.changes_by_token[pageToken])


class _FakeDriveService:
    def __init__(self, files_by_folder, file_by_id, start_token="t0"):
        self.files_by_folder = files_by_folder
        self.file_by_id = file_by_id
        self.start_token = start_token
        self.changes_by_token = {}

    def files(self):
        return _FakeFilesResource(self)

    def changes(self):
        return _FakeChangesResource(self)


def _file_meta(file_id, name=None):
    return {
        "id": file_id,
        "name": name or file_id,
        "mimeType": DOC_MIME,
        "parents": ["root"],
        "trashed": False,
        "webViewLink": f"https://drive/{file_id}",
        "modifiedTime": "2026-01-01T00:00:00Z",
        "size": None,
    }


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    from dlt.common.configuration.container import Container
    from dlt.common.pipeline import PipelineContext

    Container()[PipelineContext].deactivate()

    # add() never calls the LLM (no cognify()), but cognee's startup
    # connection check would still try to reach one — skip it so this test
    # needs no live credentials.
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / "dlt"))
    monkeypatch.setenv("PIPELINES_DIR", str(tmp_path / "dlt" / "pipelines"))

    from cognee.context_global_variables import graph_db_config, vector_db_config
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.tasks.ingestion.get_dlt_destination import get_dlt_destination

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    get_dlt_destination.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)

    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    yield

    Container()[PipelineContext].deactivate()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def _patch_drive(monkeypatch, service, content_by_id):
    monkeypatch.setattr(gd_source, "build_drive_service", lambda **kwargs: service)

    def fake_extract(service, file_id, mime_type, name):
        return content_by_id[file_id]

    monkeypatch.setattr(gd_source, "extract_file_content", fake_extract)


async def _dlt_sourced_data(dataset_name: str):
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[dataset_name]
    )
    if not datasets:
        return []
    all_data = await get_dataset_data(datasets[0].id)
    return [
        d
        for d in all_data
        if isinstance(d.system_metadata, dict) and d.system_metadata.get("source") == "google_drive"
    ]


async def _remember_drive(**overrides):
    source = google_drive_source(folder_id="root", auth_mode="service_account")
    # The connector declares its document nature on the source (DOCUMENT_SOURCE_ATTR),
    # so document-mode routing must work from a plain add()/remember() call
    # (asserted below via is_dlt_sourced).
    kwargs = {
        "dataset_name": DATASET_NAME,
        "primary_key": "id",
        "write_disposition": "merge",
        "max_rows_per_table": 0,
    }
    kwargs.update(overrides)
    await cognee.add(source, **kwargs)


@pytest.mark.asyncio
async def test_separate_folder_syncs_preserve_each_others_documents(clean_environment, monkeypatch):
    file_a = {**_file_meta("fileA"), "parents": ["folderA"]}
    file_b = {**_file_meta("fileB"), "parents": ["folderB"]}
    service = _FakeDriveService(
        files_by_folder={"folderA": [file_a], "folderB": [file_b]},
        file_by_id={"fileA": file_a, "fileB": file_b},
    )
    _patch_drive(
        monkeypatch, service, {"fileA": "Alice works at Acme.", "fileB": "Bob works at Beta."}
    )
    for folder in ("folderA", "folderB"):
        await cognee.add(
            google_drive_source(
                folder_id=folder, resource_name=f"drive_{folder.lower()}", service=service
            ),
            dataset_name=DATASET_NAME,
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )
    records = await _dlt_sourced_data(DATASET_NAME)
    assert {record.system_metadata["external_id"] for record in records} == {"fileA", "fileB"}


@pytest.mark.asyncio
async def test_deleting_the_last_file_forgets_the_document(clean_environment, monkeypatch):
    file_a = _file_meta("fileA")
    service = _FakeDriveService(files_by_folder={"root": [file_a]}, file_by_id={"fileA": file_a})
    _patch_drive(monkeypatch, service, {"fileA": "Alice works at Acme."})
    await _remember_drive()
    assert len(await _dlt_sourced_data(DATASET_NAME)) == 1
    service.changes_by_token["t0"] = {
        "changes": [],
        "newStartPageToken": "t1",
    }
    await _remember_drive()
    assert len(await _dlt_sourced_data(DATASET_NAME)) == 1
    # A failed extraction must not look like an authoritative empty snapshot.
    with pytest.raises(Exception, match="failed to list changes"):
        await _remember_drive()
    assert len(await _dlt_sourced_data(DATASET_NAME)) == 1
    service.changes_by_token["t1"] = {
        "changes": [{"fileId": "fileA", "removed": True}],
        "newStartPageToken": "t2",
    }
    await _remember_drive()
    assert await _dlt_sourced_data(DATASET_NAME) == []


@pytest.mark.asyncio
async def test_incremental_resync_and_deletion_propagate(clean_environment, monkeypatch):
    file_a = _file_meta("fileA")
    file_b = _file_meta("fileB")
    file_c = _file_meta("fileC")
    service = _FakeDriveService(
        files_by_folder={"root": [file_a, file_b, file_c]},
        file_by_id={"fileA": file_a, "fileB": file_b, "fileC": file_c},
        start_token="t0",
    )
    _patch_drive(
        monkeypatch,
        service,
        {
            "fileA": "Alice works at Acme.",
            "fileB": "Bob works at Acme.",
            "fileC": "Carol works at Acme.",
        },
    )

    await _remember_drive()

    initial_data = await _dlt_sourced_data(DATASET_NAME)
    assert len(initial_data) == 3
    initial_ids_by_file = {d.system_metadata["external_id"]: d.id for d in initial_data}
    assert set(initial_ids_by_file) == {"fileA", "fileB", "fileC"}

    # Document rows are tagged source="google_drive" (not "dlt"), so is_dlt_sourced()
    # is False and classify_documents picks TextDocument (chunk + LLM extract) over
    # the schema-wrapped DltRowDocument. If this regressed, the files would still
    # ingest but contribute nothing to the graph.
    from cognee.tasks.ingestion.dlt_utils import is_dlt_sourced

    assert all(not is_dlt_sourced(d.system_metadata) for d in initial_data)

    # --- Incremental run: fileA content changes, fileB is deleted from Drive,
    # and fileC is left untouched (not in the change feed).
    service.changes_by_token["t0"] = {
        "changes": [
            {"fileId": "fileA", "removed": False},
            {"fileId": "fileB", "removed": True},
        ],
        "newStartPageToken": "t1",
    }
    file_a["name"] = "fileA-renamed"  # still the same id/scope, just changed
    del service.file_by_id["fileB"]  # simulate permanent deletion

    _patch_drive(
        monkeypatch,
        service,
        {"fileA": "Alice now works at Globex.", "fileB": "unused", "fileC": "unused"},
    )

    await _remember_drive()

    final_data = await _dlt_sourced_data(DATASET_NAME)
    final_by_file = {d.system_metadata["external_id"]: d for d in final_data}

    # fileB was removed from Drive -> forgotten via orphan_cleanup.
    assert "fileB" not in final_by_file

    # fileA's content changed -> new content-hash-based data_id (old version
    # gone, new version present) — existing dlt versioning behavior.
    assert final_by_file["fileA"].id != initial_ids_by_file["fileA"]

    # fileC was untouched -> same data_id, not re-created.
    assert final_by_file["fileC"].id == initial_ids_by_file["fileC"]

    assert len(final_data) == 2
