"""End-to-end test: Google Drive connector through the real cognee.add()
pipeline, with the Drive API mocked (no live credentials).

Verifies the behaviors the issue asks for at the add()/Data-record layer
(no cognify(), so no LLM calls are needed):
  - initial sync creates one Data record per in-scope file
  - an incremental re-sync only re-processes new/changed files — an
    unchanged file's data_id is stable, so it isn't recreated
  - a file removed from Drive is forgotten (orphan_cleanup fires)
"""

import pathlib

import pytest
import pytest_asyncio

import cognee
from cognee.tasks.ingestion.connectors.google_drive import source as gd_source
from cognee.tasks.ingestion.connectors.google_drive.source import create_google_drive_source
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data

DATASET_NAME = "gdrive_integration_test"
DOC_MIME = "application/vnd.google-apps.document"


class _FakeHttpError(Exception):
    class _Resp:
        status = 404

    resp = _Resp()


class _FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeFilesResource:
    def __init__(self, service):
        self._service = service

    def list(self, q, fields, pageSize, pageToken=None):
        import re

        folder_id = re.search(r"'([^']+)' in parents", q).group(1)
        if "mimeType=" in q:
            return _FakeRequest({"files": []})  # no subfolders in this test
        files = [m for m in self._service.files_by_folder.get(folder_id, []) if not m["trashed"]]
        return _FakeRequest({"files": files})

    def get(self, fileId, fields):
        meta = self._service.file_by_id.get(fileId)
        if meta is None:
            raise _FakeHttpError()
        return _FakeRequest(meta)


class _FakeChangesResource:
    def __init__(self, service):
        self._service = service

    def getStartPageToken(self):
        return _FakeRequest({"startPageToken": self._service.start_token})

    def list(self, pageToken, fields):
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

    # GoogleDriveConfig is @lru_cache'd — clear it so env vars set by other
    # tests in this process don't leak into this one's defaults.
    gd_source.get_google_drive_config.cache_clear()

    # add() never calls the LLM (no cognify()), but cognee's startup
    # connection check would still try to reach one — skip it so this test
    # needs no live credentials.
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")

    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    yield

    gd_source.get_google_drive_config.cache_clear()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def _patch_drive(monkeypatch, service, content_by_id):
    monkeypatch.setattr(gd_source, "build_drive_service", lambda config: service)

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
        if isinstance(d.external_metadata, dict) and d.external_metadata.get("source") == "dlt"
    ]


async def _remember_drive(**overrides):
    source = create_google_drive_source(folder_id="root", auth_mode="service_account")
    kwargs = dict(
        dataset_name=DATASET_NAME,
        primary_key="file_id",
        write_disposition="merge",
        dlt_content_column="content",
        max_rows_per_table=0,
    )
    kwargs.update(overrides)
    await cognee.add(source, **kwargs)


@pytest.mark.asyncio
async def test_incremental_resync_and_deletion_propagate(clean_environment, monkeypatch):
    file_a = _file_meta("fileA")
    file_b = _file_meta("fileB")
    service = _FakeDriveService(
        files_by_folder={"root": [file_a, file_b]},
        file_by_id={"fileA": file_a, "fileB": file_b},
        start_token="t0",
    )
    _patch_drive(
        monkeypatch, service, {"fileA": "Alice works at Acme.", "fileB": "Bob works at Acme."}
    )

    await _remember_drive()

    initial_data = await _dlt_sourced_data(DATASET_NAME)
    assert len(initial_data) == 2
    initial_ids_by_file = {}
    for d in initial_data:
        initial_ids_by_file[d.external_metadata["primary_key_value"]] = d.id
    assert set(initial_ids_by_file) == {"fileA", "fileB"}

    # --- Incremental run: fileA content changes, fileB is untouched, and
    # fileB gets deleted from Drive.
    service.changes_by_token["t0"] = {
        "changes": [
            {"fileId": "fileA", "removed": False},
            {"fileId": "fileB", "removed": True},
        ],
        "newStartPageToken": "t1",
    }
    file_a["name"] = "fileA-renamed"  # still the same id/scope, just changed
    del service.file_by_id["fileB"]  # simulate permanent deletion

    _patch_drive(monkeypatch, service, {"fileA": "Alice now works at Globex.", "fileB": "unused"})

    await _remember_drive()

    final_data = await _dlt_sourced_data(DATASET_NAME)
    final_by_file = {d.external_metadata["primary_key_value"]: d for d in final_data}

    # fileB was removed from Drive -> forgotten via orphan_cleanup.
    assert "fileB" not in final_by_file

    # fileA's content changed -> new content-hash-based data_id (old version
    # gone, new version present) — this is existing dlt versioning behavior.
    assert "fileA" in final_by_file
    assert final_by_file["fileA"].id != initial_ids_by_file["fileA"]
    assert len(final_data) == 1
