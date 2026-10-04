# Adapted from topoteretes/cognee-community (Apache-2.0), commit 21d0b36.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
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

DOC_MIME = "application/vnd.google-apps.document"
PDF_MIME = "application/pdf"


def test_refuses_core_without_table_scoped_cleanup(monkeypatch):
    pytest.importorskip("dlt")
    monkeypatch.setattr(gd_source.dlt_utils, "DOCUMENT_SYNC_VERSION", 0)
    with pytest.raises(RuntimeError, match="table-scoped"):
        gd_source.google_drive_source(folder_id="root", service=object())


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

    def list(
        self,
        q,
        fields,
        pageSize,
        pageToken=None,
        *,
        supportsAllDrives,
        includeItemsFromAllDrives,
        corpora=None,
        driveId=None,
    ):
        assert supportsAllDrives is True
        assert includeItemsFromAllDrives is True
        self._service.list_calls.append(q)
        if driveId:
            assert corpora == "drive"
            return _FakeRequest(
                {
                    "files": [
                        meta
                        for meta in self._service.file_by_id.values()
                        if meta.get("driveId") == driveId and not meta.get("trashed")
                    ]
                }
            )
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

    def get(self, fileId, fields, supportsAllDrives):
        assert supportsAllDrives is True
        self._service.get_calls.append(fileId)
        meta = self._service.file_by_id.get(fileId)
        if meta is None:
            return _FakeRequest(error=FakeHttpError(404))
        return _FakeRequest(meta)


class _FakeChangesResource:
    def __init__(self, service):
        self._service = service

    def getStartPageToken(self, supportsAllDrives, driveId=None):
        assert supportsAllDrives is True
        self._service.start_drive_ids.append(driveId)
        return _FakeRequest({"startPageToken": self._service.start_token})

    def list(self, pageToken, fields, supportsAllDrives, includeItemsFromAllDrives, driveId=None):
        assert supportsAllDrives is True
        assert includeItemsFromAllDrives is True
        self._service.change_drive_ids.append(driveId)
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
        self.start_drive_ids = []
        self.change_drive_ids = []

    def files(self):
        return _FakeFilesResource(self)

    def changes(self):
        return _FakeChangesResource(self)


def _config(**overrides):
    defaults = {
        "folder_id": "root",
        "auth_mode": "service_account",
        "credentials_path": "unused.json",
        "token_path": None,
        "include_subfolders": True,
        "max_file_size_mb": 25,
    }
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

    assert {row["id"] for row in rows} == {"fileA", "fileB"}
    assert all(row["_deleted"] is False for row in rows)
    assert sorted(fake_content_extraction) == ["fileA", "fileB"]
    assert service.get_calls == []  # initial sync only lists, never GETs individually
    # The changes cursor is captured before listing, then persisted for the
    # next incremental run.
    assert state["page_token"] == "t0"


def test_full_scan_deletes_absent_files_despite_permanently_bad_content(monkeypatch):
    good, bad = _file_meta("good"), _file_meta("bad", mime_type=PDF_MIME)
    service = FakeDriveService({"root": [good, bad]}, {"good": good, "bad": bad})
    monkeypatch.setattr(
        gd_source,
        "extract_file_content",
        lambda service, file_id, *args: "content" if file_id == "good" else None,
    )
    state = {}
    assert [row["id"] for row in gd_source._iter_rows(service, _config(), state)] == ["good"]
    assert "page_token" not in state  # bad PDF still requires a retry
    service.files_by_folder["root"] = [bad]
    assert list(gd_source._iter_rows(service, _config(), state)) == [
        {"id": "good", "_deleted": True}
    ]
    assert state["known_ids"] == ["bad"]  # failed extraction is not deletion


def test_incomplete_full_listing_does_not_reconcile_absence(monkeypatch):
    service = FakeDriveService({}, {})
    state = {"known_ids": ["keep"]}

    def failed_listing(*args):
        yield _file_meta("new")
        raise RuntimeError("listing failed on page two")

    monkeypatch.setattr(gd_source, "_list_files_in_scope", failed_listing)
    rows = gd_source._iter_rows(service, _config(), state)
    assert next(rows)["id"] == "new"
    with pytest.raises(RuntimeError):
        next(rows)
    assert state == {"known_ids": ["keep"]}


def test_empty_replacement_clears_only_retired_folder_and_reselection_backfills(tmp_path):
    from cognee.modules.integrations.google.ingestion import empty_resource

    dlt = pytest.importorskip("dlt")
    pipeline = dlt.pipeline(
        pipeline_name="drive_deselection",
        dataset_name="drive",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'drive.db'}"),
        pipelines_dir=str(tmp_path / "pipelines"),
    )
    a, b = _file_meta("a"), _file_meta("b", parents=("other",))
    service = FakeDriveService({"root": [a], "other": [b]}, {"a": a, "b": b})

    def sync(folder, name):
        pipeline.run(
            gd_source.google_drive_source(folder_id=folder, resource_name=name, service=service)
        )

    sync("root", "folder_a")
    sync("other", "folder_b")
    pipeline.run(empty_resource("google_drive", "folder_a"), write_disposition="replace")
    with pipeline.sql_client() as sql:
        assert sql.execute_sql("SELECT id FROM folder_a") == []
        assert sql.execute_sql("SELECT id FROM folder_b") == [("b",)]
    sync("root", "folder_a")  # no Changes response configured: must full-scan again
    with pipeline.sql_client() as sql:
        assert sql.execute_sql("SELECT id FROM folder_a") == [("a",)]


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

    assert [row["id"] for row in rows] == ["fileA"]
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

    assert rows == [{"id": "fileB", "_deleted": True}]
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

    assert rows == [{"id": "fileA", "_deleted": True}]
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

    assert rows == [{"id": "fileA", "_deleted": True}]


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

    assert rows == [{"id": "fileB", "_deleted": True}]
    # No files().list() calls at all: deletions don't consult the scope set.
    assert service.list_calls == []


def test_drive_api_error_raises_instead_of_deleting_everything(fake_content_extraction):
    class BrokenChangesService(FakeDriveService):
        def changes(self):
            class _Broken:
                def list(self, pageToken, fields, **kwargs):
                    return _FakeRequest(error=RuntimeError("simulated network failure"))

            return _Broken()

    service = BrokenChangesService(files_by_folder={"root": []}, file_by_id={})
    state = {"page_token": "t0"}

    with pytest.raises(RuntimeError):
        list(gd_source._iter_rows(service, _config(), state))

    # State must be untouched — a failed run must not look like "everything
    # was deleted" to the caller (dlt then rolls back and retries this token).
    assert state == {"page_token": "t0"}


def test_shared_drive_backfill_and_incremental_changes_use_the_same_corpus():
    shared = dict(_file_meta("shared-file", parents=("nested",)), driveId="team-drive")
    other = dict(_file_meta("other-file"), driveId="other-drive")
    service = FakeDriveService(
        files_by_folder={},
        file_by_id={"shared-file": shared, "other-file": other},
        start_token="t0",
    )
    config = _config(folder_id="team-drive", shared_drive_id="team-drive")
    state = {}
    assert [row["id"] for row in gd_source._iter_rows(service, config, state)] == ["shared-file"]
    assert service.start_drive_ids == ["team-drive"]
    assert service.list_calls == ["trashed=false"]

    service.changes_by_token["t0"] = {
        "changes": [{"fileId": "shared-file"}],
        "newStartPageToken": "t1",
    }
    updated = list(gd_source._iter_rows(service, config, state))
    assert updated[0]["id"] == "shared-file"
    assert updated[0]["_deleted"] is False
    assert service.change_drive_ids == ["team-drive"]

    service.changes_by_token["t1"] = {
        "changes": [{"fileId": "shared-file", "removed": True}],
        "newStartPageToken": "t2",
    }
    assert list(gd_source._iter_rows(service, config, state)) == [
        {"id": "shared-file", "_deleted": True}
    ]
    assert state["page_token"] == "t2"


def test_drive_membership_events_without_a_file_id_do_not_break_sync():
    service = FakeDriveService(files_by_folder={}, file_by_id={})
    service.changes_by_token["t0"] = {
        "changes": [{"removed": True}],
        "newStartPageToken": "t1",
    }
    state = {"page_token": "t0"}
    assert list(gd_source._iter_rows(service, _config(), state)) == []
    assert state["page_token"] == "t1"


def test_file_diagnostics_report_reasons_and_retry_failed_extraction(monkeypatch):
    files = [
        _file_meta("good"),
        _file_meta("unsupported", mime_type="image/png"),
        dict(_file_meta("large"), size=str(30 * 1024 * 1024)),
        _file_meta("empty"),
        _file_meta("failed"),
    ]
    service = FakeDriveService(
        files_by_folder={"root": files},
        file_by_id={item["id"]: item for item in files},
        start_token="t0",
    )
    content = {"good": "hello", "empty": " ", "failed": None}
    monkeypatch.setattr(gd_source, "extract_file_content", lambda _, fid, *_args: content[fid])
    state, stats = {}, {}
    rows = list(gd_source._iter_rows(service, _config(), state, stats))
    assert [row["id"] for row in rows] == ["good"]
    assert stats == {
        "scanned": 5,
        "skipped": 3,
        "failed": 1,
        "deleted": 0,
        "skipped_unsupported_type": 1,
        "skipped_too_large": 1,
        "skipped_empty_content": 1,
        "failed_content_extraction": 1,
    }
    assert "page_token" not in state

    content["failed"] = "recovered"
    assert {row["id"] for row in gd_source._iter_rows(service, _config(), state, stats)} == {
        "good",
        "failed",
    }
    assert stats["failed"] == 0
    assert stats["scanned"] == 5
    assert state["page_token"] == "t0"
