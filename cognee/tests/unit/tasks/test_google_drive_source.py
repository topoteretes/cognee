"""Unit tests for the Google Drive dlt source's sync logic.

Exercises ``_iter_rows`` directly against a fake Drive API service (no
network, no live credentials) — covers initial full sync, incremental
syncs that only touch changed files, and deletion propagation via the
merge hard-delete tombstone row. ``build_drive_service``/auth and content
extraction (Docs/Sheets/PDF export) are monkeypatched out here so these
tests stay focused on the sync state machine.
"""

import re

import pytest

from cognee.tasks.ingestion.connectors import google_drive as gd_source
from cognee.tasks.ingestion.connectors.google_drive import _DriveConfig

FOLDER_MIME = "application/vnd.google-apps.folder"
DOC_MIME = "application/vnd.google-apps.document"
PDF_MIME = "application/pdf"


class FakeHttpError(Exception):
    class _Resp:
        def __init__(self, status):
            self.status = status

    def __init__(self, status):
        self.resp = self._Resp(status)


class _FakeRequest:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def execute(self):
        # Real googleapiclient raises HttpError from .execute(), not from the
        # request-builder call — mirror that here.
        if self._error is not None:
            raise self._error
        return self._result


class _FakeFilesResource:
    def __init__(self, service):
        self._service = service

    def list(self, q, fields, pageSize, pageToken=None):
        self._service.list_calls.append(q)
        match = re.search(r"'([^']+)' in parents", q)
        folder_id = match.group(1)
        if "mimeType=" in q:
            children = self._service.subfolders.get(folder_id, [])
            return _FakeRequest({"files": [{"id": fid} for fid in children]})
        files = [
            meta
            for meta in self._service.files_by_folder.get(folder_id, [])
            if not meta.get("trashed")
        ]
        return _FakeRequest({"files": files})

    def get(self, fileId, fields):
        self._service.get_calls.append(fileId)
        meta = self._service.file_by_id.get(fileId)
        if meta is None:
            return _FakeRequest(error=FakeHttpError(404))
        return _FakeRequest(meta)


class _FakeChangesResource:
    def __init__(self, service):
        self._service = service

    def getStartPageToken(self):
        return _FakeRequest({"startPageToken": self._service.start_token})

    def list(self, pageToken, fields):
        return _FakeRequest(self._service.changes_by_token[pageToken])


class FakeDriveService:
    def __init__(self, files_by_folder, file_by_id, start_token="token-0", subfolders=None):
        self.files_by_folder = files_by_folder
        self.file_by_id = file_by_id
        self.start_token = start_token
        self.subfolders = subfolders or {}
        self.changes_by_token = {}
        self.get_calls = []
        self.list_calls = []

    def files(self):
        return _FakeFilesResource(self)

    def changes(self):
        return _FakeChangesResource(self)


def _config(**overrides):
    defaults = dict(
        folder_id="root",
        auth_mode="service_account",
        credentials_path="unused.json",
        token_path=None,
        include_subfolders=True,
        max_file_size_mb=25,
    )
    defaults.update(overrides)
    return _DriveConfig(**defaults)


@pytest.fixture(autouse=True)
def fake_content_extraction(monkeypatch):
    calls = []

    def fake_extract(service, file_id, mime_type, name):
        calls.append(file_id)
        return f"content::{file_id}"

    monkeypatch.setattr(gd_source, "extract_file_content", fake_extract)
    return calls


def _file_meta(file_id, mime_type=DOC_MIME, parents=("root",), trashed=False):
    return {
        "id": file_id,
        "name": file_id,
        "mimeType": mime_type,
        "parents": list(parents),
        "trashed": trashed,
        "webViewLink": f"https://drive/{file_id}",
        "modifiedTime": "2026-01-01T00:00:00Z",
    }


def test_initial_sync_yields_all_files_and_advances_state(fake_content_extraction):
    file_a = _file_meta("fileA", mime_type=DOC_MIME)
    file_b = _file_meta("fileB", mime_type=PDF_MIME)
    service = FakeDriveService(
        files_by_folder={"root": [file_a, file_b]},
        file_by_id={"fileA": file_a, "fileB": file_b},
        start_token="t0",
    )
    state = {}

    rows = list(gd_source._iter_rows(service, _config(), state))

    assert {row["file_id"] for row in rows} == {"fileA", "fileB"}
    assert all(row["deleted"] is False for row in rows)
    assert sorted(fake_content_extraction) == ["fileA", "fileB"]
    assert service.get_calls == []  # initial sync only lists, never GETs individually
    # The changes cursor is captured before listing, then persisted for the
    # next incremental run.
    assert state["page_token"] == "t0"


def test_incremental_sync_only_touches_changed_file(fake_content_extraction):
    file_a = _file_meta("fileA")
    file_b = _file_meta("fileB")
    service = FakeDriveService(
        files_by_folder={"root": [file_a, file_b]},
        file_by_id={"fileA": file_a, "fileB": file_b},
    )
    service.changes_by_token["t0"] = {
        "changes": [{"fileId": "fileA", "removed": False}],
        "newStartPageToken": "t1",
    }
    state = {"page_token": "t0"}

    rows = list(gd_source._iter_rows(service, _config(), state))

    assert [row["file_id"] for row in rows] == ["fileA"]
    assert service.get_calls == ["fileA"]
    assert fake_content_extraction == ["fileA"]  # fileB was never re-extracted
    assert state["page_token"] == "t1"


def test_removed_file_yields_hard_delete_tombstone(fake_content_extraction):
    file_a = _file_meta("fileA")
    service = FakeDriveService(
        files_by_folder={"root": [file_a]},
        file_by_id={"fileA": file_a},
    )
    service.changes_by_token["t1"] = {
        "changes": [{"fileId": "fileB", "removed": True}],
        "newStartPageToken": "t2",
    }
    state = {"page_token": "t1"}

    rows = list(gd_source._iter_rows(service, _config(), state))

    assert rows == [{"file_id": "fileB", "deleted": True}]
    assert service.get_calls == []  # "removed" changes need no metadata fetch
    assert fake_content_extraction == []
    assert state["page_token"] == "t2"


def test_file_moved_out_of_scope_yields_hard_delete_tombstone(fake_content_extraction):
    # fileA still exists and isn't trashed, but its parent is no longer the
    # configured folder — e.g. the user moved it elsewhere in Drive.
    file_a = _file_meta("fileA", parents=("some_other_folder",))
    service = FakeDriveService(
        files_by_folder={"root": []},
        file_by_id={"fileA": file_a},
    )
    service.changes_by_token["t0"] = {
        "changes": [{"fileId": "fileA", "removed": False}],
        "newStartPageToken": "t1",
    }
    state = {"page_token": "t0"}

    rows = list(gd_source._iter_rows(service, _config(), state))

    assert rows == [{"file_id": "fileA", "deleted": True}]
    assert fake_content_extraction == []


def test_deleted_file_returning_404_on_get_yields_tombstone(fake_content_extraction):
    # Changes API reports a change but the file is already gone by the time
    # we fetch its metadata.
    service = FakeDriveService(files_by_folder={"root": []}, file_by_id={})
    service.changes_by_token["t0"] = {
        "changes": [{"fileId": "fileA", "removed": False}],
        "newStartPageToken": "t1",
    }
    state = {"page_token": "t0"}

    rows = list(gd_source._iter_rows(service, _config(), state))

    assert rows == [{"file_id": "fileA", "deleted": True}]


def test_only_deletions_skips_the_subfolder_scope_walk(fake_content_extraction):
    # When a change page carries only deletions, there is no need to walk the
    # folder tree to build the scope set — that work only scope-checks changed
    # files.
    service = FakeDriveService(files_by_folder={"root": []}, file_by_id={})
    service.changes_by_token["t0"] = {
        "changes": [{"fileId": "fileB", "removed": True}],
        "newStartPageToken": "t1",
    }
    state = {"page_token": "t0"}

    rows = list(gd_source._iter_rows(service, _config(), state))

    assert rows == [{"file_id": "fileB", "deleted": True}]
    # No files().list() calls at all: deletions don't consult the scope set.
    assert service.list_calls == []


def test_drive_api_error_raises_instead_of_deleting_everything(fake_content_extraction):
    class BrokenChangesService(FakeDriveService):
        def changes(self):
            class _Broken:
                def list(self, pageToken, fields):
                    return _FakeRequest(error=RuntimeError("simulated network failure"))

            return _Broken()

    service = BrokenChangesService(files_by_folder={"root": []}, file_by_id={})
    state = {"page_token": "t0"}

    with pytest.raises(RuntimeError):
        list(gd_source._iter_rows(service, _config(), state))

    # State must be untouched — a failed run must not look like "everything
    # was deleted" to the caller (dlt then rolls back and retries this token).
    assert state == {"page_token": "t0"}
