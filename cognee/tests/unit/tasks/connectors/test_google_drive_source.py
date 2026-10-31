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

import drive_pdfs
import pytest

from cognee.tasks.ingestion.connectors import google_drive as gd_source
from cognee.tasks.ingestion.connectors.google_drive import _DriveConfig

_REAL_EXTRACT = gd_source.extract_file_content  # the autouse fixture replaces it per test

DOC_MIME = "application/vnd.google-apps.document"
PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


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


EXECUTE_RETRIES: list[int] = []


class _FakeRequest:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def execute(self, num_retries=0):
        EXECUTE_RETRIES.append(num_retries)
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
        page_size = self._service.page_size
        if page_size:
            start = int(pageToken or 0)
            response = {"files": files[start : start + page_size]}
            if start + page_size < len(files):
                response["nextPageToken"] = str(start + page_size)
            return _FakeRequest(response)
        return _FakeRequest({"files": files})

    def get(self, fileId, fields, supportsAllDrives):
        assert supportsAllDrives is True
        if fields == "id":  # resolving a folder, not fetching a file
            real_id = self._service.folder_ids.get(fileId, fileId)
            return _FakeRequest({"id": real_id})
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
        self.folder_ids = {"root": "0Aroot"}  # alias -> the real id Drive reports in parents
        self.list_calls = []
        self.start_drive_ids = []
        self.change_drive_ids = []
        self.closed = False
        self.page_size = None  # set to make files.list paginate with nextPageToken

    def files(self):
        return _FakeFilesResource(self)

    def changes(self):
        return _FakeChangesResource(self)

    def close(self):
        self.closed = True


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


def _file_meta(file_id, mime_type=DOC_MIME, parents=("0Aroot",), trashed=False):
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

    shared["modifiedTime"] = "2026-02-01T00:00:00Z"
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
    rows = list(gd_source._iter_rows(service, _config(max_file_size_mb=25), state, stats))
    assert [row["id"] for row in rows] == ["good"]  # nothing earlier to forget: no tombstones
    assert stats == {
        "scanned": 5,
        "skipped": 3,
        "failed": 1,
        "deleted": 0,
        "unchanged": 0,
        "skipped_unsupported_type": 1,
        "skipped_too_large": 1,
        "skipped_empty_content": 1,
        "failed_content_extraction": 1,
    }
    assert "page_token" not in state

    content["failed"] = "recovered"
    # Only "good" produced a row, so only it is skipped as fetched; the failed one is retried.
    config = _config(max_file_size_mb=25)
    assert [row["id"] for row in gd_source._iter_rows(service, config, state, stats)] == ["failed"]
    assert stats["failed"] == 0
    assert stats["scanned"] == 5
    assert stats["unchanged"] == 1
    assert state["page_token"] == "t0"


def _many_files(count):
    files = [_file_meta(f"f{index:03d}") for index in range(count)]
    return files, FakeDriveService({"root": files}, {item["id"]: item for item in files})


def test_parallel_fetch_keeps_order_and_gives_each_thread_its_own_client(monkeypatch):
    import threading

    files, service = _many_files(40)
    clients, seen = [], set()
    lock = threading.Lock()
    # Two fetches must be in flight together for either to finish: proves real concurrency.
    barrier = threading.Barrier(2, timeout=10)
    first_two = threading.Semaphore(2)

    def factory():
        client = FakeDriveService(service.files_by_folder, service.file_by_id)
        with lock:
            clients.append(client)
        return client

    def fake_extract(client, file_id, mime_type, name):
        with lock:
            seen.add((id(client), threading.get_ident()))
        if first_two.acquire(blocking=False):
            barrier.wait()
        return f"content::{file_id}"

    monkeypatch.setattr(gd_source, "extract_file_content", fake_extract)
    state = {}
    rows = list(
        gd_source._iter_rows(service, _config(max_workers=4), state, service_factory=factory)
    )

    assert [row["id"] for row in rows] == [item["id"] for item in files]
    assert 2 <= len(clients) <= 4
    assert len({thread for _, thread in seen}) == len({client for client, _ in seen})
    assert all(client.closed for client in clients)
    assert state["page_token"] == "token-0"
    assert set(state["versions"]) == {item["id"] for item in files}


def test_parallel_fetch_without_factory_runs_inline_on_the_given_service(monkeypatch):
    import threading

    _, service = _many_files(5)
    threads, clients = set(), set()

    def fake_extract(client, file_id, *_args):
        threads.add(threading.get_ident())
        clients.add(id(client))
        return "x"

    monkeypatch.setattr(gd_source, "extract_file_content", fake_extract)
    list(gd_source._iter_rows(service, _config(max_workers=4), {}))
    assert threads == {threading.get_ident()}
    assert clients == {id(service)}


def test_failure_in_one_worker_surfaces_and_stops_the_rest(monkeypatch):
    _, service = _many_files(30)
    started = []

    def fake_extract(client, file_id, *_args):
        started.append(file_id)
        if file_id == "f001":
            raise AssertionError("boom")  # not an extraction failure: must surface
        return "x"

    monkeypatch.setattr(gd_source, "extract_file_content", fake_extract)
    # extract_file_content swallowing is bypassed by the fake, so the error escapes.
    with pytest.raises(AssertionError):
        list(
            gd_source._iter_rows(
                service, _config(max_workers=2), {}, service_factory=lambda: service
            )
        )
    assert len(started) < 30


def test_full_rescan_after_failure_refetches_only_changed_and_failed_files(monkeypatch):
    _, service = _many_files(4)
    fetched = []
    broken = {"f003"}

    def fake_extract(client, file_id, *_args):
        fetched.append(file_id)
        return None if file_id in broken else "x"

    monkeypatch.setattr(gd_source, "extract_file_content", fake_extract)
    state = {}
    list(gd_source._iter_rows(service, _config(), state))
    assert "page_token" not in state
    assert sorted(state["versions"]) == ["f000", "f001", "f002"]

    fetched.clear()
    broken.clear()
    service.file_by_id["f001"]["modifiedTime"] = "2026-02-01T00:00:00Z"  # edited meanwhile
    list(gd_source._iter_rows(service, _config(), state))
    assert sorted(fetched) == ["f001", "f003"]
    assert state["page_token"] == "token-0"


def test_changed_file_at_a_fetched_version_is_not_downloaded_again(fake_content_extraction):
    file_a = _file_meta("fileA")
    service = FakeDriveService({"root": [file_a]}, {"fileA": file_a})
    service.changes_by_token["t0"] = {
        "changes": [{"fileId": "fileA", "removed": False}],
        "newStartPageToken": "t1",
    }
    state = {"page_token": "t0", "versions": {"fileA": file_a["modifiedTime"]}}
    assert list(gd_source._iter_rows(service, _config(), state)) == []
    assert fake_content_extraction == []
    assert state["page_token"] == "t1"


def test_deleted_file_forgets_its_version(fake_content_extraction):
    service = FakeDriveService({"root": []}, {})
    service.changes_by_token["t0"] = {
        "changes": [{"fileId": "fileA", "removed": True}],
        "newStartPageToken": "t1",
    }
    state = {"page_token": "t0", "versions": {"fileA": "v", "fileB": "v"}}
    list(gd_source._iter_rows(service, _config(), state))
    assert state["versions"] == {"fileB": "v"}


def test_max_workers_must_be_positive():
    pytest.importorskip("dlt")
    with pytest.raises(ValueError, match="max_workers"):
        gd_source.google_drive_source(folder_id="root", service=object(), max_workers=0)


def test_no_size_limit_by_default(monkeypatch):
    monkeypatch.delenv("GOOGLE_DRIVE_MAX_FILE_SIZE_MB", raising=False)
    assert gd_source._env_int("GOOGLE_DRIVE_MAX_FILE_SIZE_MB") is None
    monkeypatch.setenv("GOOGLE_DRIVE_MAX_FILE_SIZE_MB", "10")
    assert gd_source._env_int("GOOGLE_DRIVE_MAX_FILE_SIZE_MB") == 10
    monkeypatch.setenv("GOOGLE_DRIVE_MAX_FILE_SIZE_MB", "ten")
    with pytest.raises(ValueError, match="GOOGLE_DRIVE_MAX_FILE_SIZE_MB"):
        gd_source._env_int("GOOGLE_DRIVE_MAX_FILE_SIZE_MB")
    huge = dict(_file_meta("huge"), size=str(5 * 1024 * 1024 * 1024))
    service = FakeDriveService({"root": [huge]}, {"huge": huge})
    config = _config(max_file_size_mb=None)
    assert [row["id"] for row in gd_source._iter_rows(service, config, {})] == ["huge"]


def test_raised_size_limit_fetches_the_file_again(fake_content_extraction):
    big = dict(_file_meta("big"), size=str(30 * 1024 * 1024))
    service = FakeDriveService({"root": [big]}, {"big": big})
    state = {}
    first = list(gd_source._iter_rows(service, _config(max_file_size_mb=25), state))
    assert first == []  # never ingested, so there is nothing to forget
    assert fake_content_extraction == []
    assert "big" not in state["versions"]
    state.pop("page_token")
    second = list(gd_source._iter_rows(service, _config(max_file_size_mb=None), state))
    assert [row["id"] for row in second] == ["big"]
    assert second[0]["_deleted"] is False


def test_changed_file_grown_over_the_limit_is_forgotten(fake_content_extraction):
    big = dict(_file_meta("big"), size=str(30 * 1024 * 1024))
    service = FakeDriveService({"root": [big]}, {"big": big})
    service.changes_by_token["t0"] = {
        "changes": [{"fileId": "big", "removed": False}],
        "newStartPageToken": "t1",
    }
    state = {"page_token": "t0", "known_ids": ["big"], "versions": {"big": "old"}}
    rows = list(gd_source._iter_rows(service, _config(max_file_size_mb=25), state))
    assert rows == [{"id": "big", "_deleted": True}]
    assert "big" not in state["versions"]
    assert state["page_token"] == "t1"


def test_export_over_the_google_limit_fails_the_run_with_its_reason(monkeypatch):
    doc = _file_meta("doc")
    service = FakeDriveService({"root": [doc]}, {"doc": doc})

    def too_large(*_args):
        raise gd_source.ExportTooLargeError("doc")

    monkeypatch.setattr(gd_source, "extract_file_content", too_large)
    state, stats = {}, {}
    assert list(gd_source._iter_rows(service, _config(), state, stats)) == []
    assert stats["failed"] == 1
    assert stats["failed_export_too_large"] == 1
    assert "failed_content_extraction" not in stats
    assert state["page_token"] == "token-0"  # no rerun can cure it, so it holds nothing back
    assert "doc" not in state["versions"]


def test_every_api_call_retries(fake_content_extraction):
    EXECUTE_RETRIES.clear()
    file_a = _file_meta("fileA")
    service = FakeDriveService({"root": [file_a]}, {"fileA": file_a}, start_token="t0")
    state = {}
    list(gd_source._iter_rows(service, _config(), state))
    file_a["modifiedTime"] = "2026-03-01T00:00:00Z"
    service.changes_by_token["t0"] = {
        "changes": [{"fileId": "fileA", "removed": False}],
        "newStartPageToken": "t1",
    }
    list(gd_source._iter_rows(service, _config(), state))
    assert EXECUTE_RETRIES
    assert set(EXECUTE_RETRIES) == {gd_source.API_RETRIES}


def test_incomplete_full_listing_does_not_reconcile_absence_on_the_threaded_path(monkeypatch):
    service = FakeDriveService({}, {})
    state = {"known_ids": ["keep"], "versions": {"keep": "v"}}

    def failed_listing(*args):
        for index in range(5):
            yield _file_meta(f"new{index}")
        raise RuntimeError("listing failed on page two")

    monkeypatch.setattr(gd_source, "_list_files_in_scope", failed_listing)
    rows = gd_source._iter_rows(
        service, _config(max_workers=2), state, service_factory=lambda: service
    )
    with pytest.raises(RuntimeError, match="page two"):
        list(rows)
    assert state == {"known_ids": ["keep"], "versions": {"keep": "v"}}


def test_withdrawn_authorization_stops_fetching_within_the_worker_count(monkeypatch):
    # Eight workers: a window of twice the workers would fetch up to sixteen files ahead.
    from cognee.tasks.ingestion import dlt_utils

    _, service = _many_files(40)
    fetched = []
    monkeypatch.setattr(
        gd_source, "extract_file_content", lambda _c, file_id, *_a: fetched.append(file_id) or "x"
    )
    checks = []

    def check_active():
        checks.append(1)
        if len(checks) > 7:
            raise PermissionError("credential revoked")

    rows = []
    rows_iter = dlt_utils.guarded_rows(
        gd_source._iter_rows(service, _config(max_workers=8), {}, service_factory=lambda: service),
        check_active,
    )
    with pytest.raises(PermissionError):
        rows.extend(rows_iter)
    assert len(rows) == 3
    assert len(fetched) <= len(rows) + 8


def test_permanent_failures_do_not_hold_the_cursor_back(monkeypatch):
    files = [_file_meta("ok"), _file_meta("word"), _file_meta("huge_doc")]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})

    def extract(_client, file_id, *_args):
        if file_id == "word":
            raise gd_source.LoaderUnavailableError("no loader")
        if file_id == "huge_doc":
            raise gd_source.ExportTooLargeError(file_id)
        return "x"

    monkeypatch.setattr(gd_source, "extract_file_content", extract)
    state, stats = {}, {}
    rows = list(gd_source._iter_rows(service, _config(), state, stats))
    assert [row["id"] for row in rows] == ["ok"]
    assert stats["failed"] == 2
    assert stats["failed_loader_unavailable"] == 1
    assert stats["failed_export_too_large"] == 1
    assert state["page_token"] == "token-0"
    assert sorted(state["versions"]) == ["ok"]

    # A transient failure alongside them still holds the cursor.
    def flaky(_client, file_id, *_args):
        return None if file_id == "ok" else extract(_client, file_id)

    monkeypatch.setattr(gd_source, "extract_file_content", flaky)
    state = {}
    list(gd_source._iter_rows(service, _config(), state, {}))
    assert "page_token" not in state


def test_in_flight_fetches_are_bounded_by_size_and_a_heavy_file_still_runs(monkeypatch):
    import threading
    import time

    monkeypatch.setattr(gd_source, "INFLIGHT_FETCH_BYTES", 100)
    running, peak = [0], [0]
    lock = threading.Lock()

    def fn(_client, item):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.01)
        with lock:
            running[0] -= 1
        return item

    items = [60, 60, 60, 500, 10, 10]
    out = list(
        gd_source._ordered_map(
            fn, items, None, lambda: FakeDriveService({}, {}), 4, weight=lambda size: size
        )
    )
    # (60 + 60 > 100), and the 500 byte item runs although it alone is over the bound
    assert [item for item, _ in out] == items
    assert peak[0] <= 2  # only the two 10 byte items can overlap with another


def test_built_client_is_closed_when_the_source_ends(monkeypatch, tmp_path):
    dlt = pytest.importorskip("dlt")
    file_a = _file_meta("a")
    built = []

    def fake_build(_credentials):
        client = FakeDriveService({"root": [file_a]}, {"a": file_a})
        built.append(client)
        return client

    monkeypatch.setattr(gd_source, "build_drive_service", lambda **_kw: fake_build(None))
    pipeline = dlt.pipeline(
        pipeline_name="drive_close",
        dataset_name="drive",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'drive.db'}"),
        pipelines_dir=str(tmp_path / "pipelines"),
    )
    pipeline.run(gd_source.google_drive_source(folder_id="root", max_workers=2))
    assert len(built) >= 2  # the listing client and one per fetch thread
    assert all(client.closed for client in built)


def test_file_directly_in_my_drive_stays_in_scope_when_drive_reports_the_real_root_id(
    fake_content_extraction,
):
    # Drive lists parents by the real id ("0Aroot"), never by the "root" alias the scope holds.
    top = dict(_file_meta("top", parents=("0Aroot",)))
    service = FakeDriveService({"root": [top]}, {"top": top}, start_token="t0")
    service.changes_by_token["t0"] = {
        "changes": [{"fileId": "top", "removed": False}],
        "newStartPageToken": "t1",
    }
    top["modifiedTime"] = "2026-05-01T00:00:00Z"
    state = {"page_token": "t0", "known_ids": ["top"], "versions": {"top": "old"}}
    rows = list(gd_source._iter_rows(service, _config(folder_id="root"), state))
    assert rows == [
        {
            "id": "top",
            "title": "top",
            "content": "content::top",
            "url": "https://drive/top",
            "_deleted": False,
        }
    ]
    assert state["known_ids"] == ["top"]


def _feed(*changes):
    return {"changes": list(changes), "newStartPageToken": "t1"}


def _trashed(file_id, trashed):
    return {"fileId": file_id, "removed": False, "file": {"trashed": trashed}}


def test_file_trashed_then_restored_in_one_interval_is_kept(fake_content_extraction):
    file_a = _file_meta("fileA")
    service = FakeDriveService({"root": [file_a]}, {"fileA": file_a}, start_token="t0")
    service.changes_by_token["t0"] = _feed(_trashed("fileA", True), _trashed("fileA", False))
    file_a["modifiedTime"] = "2026-06-01T00:00:00Z"
    state = {"page_token": "t0", "known_ids": ["fileA"], "versions": {"fileA": "old"}}
    rows = list(gd_source._iter_rows(service, _config(), state))
    assert [(row["id"], row["_deleted"]) for row in rows] == [("fileA", False)]
    assert state["known_ids"] == ["fileA"]


def test_file_restored_then_trashed_in_one_interval_is_forgotten(fake_content_extraction):
    file_a = _file_meta("fileA")
    service = FakeDriveService({"root": [file_a]}, {"fileA": file_a}, start_token="t0")
    service.changes_by_token["t0"] = _feed(_trashed("fileA", False), _trashed("fileA", True))
    state = {"page_token": "t0", "known_ids": ["fileA"], "versions": {"fileA": "old"}}
    rows = list(gd_source._iter_rows(service, _config(), state))
    assert rows == [{"id": "fileA", "_deleted": True}]
    assert state["known_ids"] == []
    assert fake_content_extraction == []


def test_feed_order_decides_across_pages_and_removal(fake_content_extraction):
    file_a = _file_meta("fileA")
    service = FakeDriveService({"root": [file_a]}, {"fileA": file_a}, start_token="t0")
    removed = {"fileId": "fileA", "removed": True}
    live = {"fileId": "fileA", "removed": False}
    page_one = {"changes": [removed], "nextPageToken": "p2"}
    page_two = {"changes": [live], "newStartPageToken": "t1"}
    service.changes_by_token["t0"] = page_one
    service.changes_by_token["p2"] = page_two
    changed, deleted, token = gd_source._list_changed_file_ids(service, "t0")
    assert (changed, deleted, token) == ({"fileA"}, set(), "t1")


class _HttpError(Exception):
    """What googleapiclient raises: a status on ``resp`` and the body in the message."""

    class _Resp:
        def __init__(self, status):
            self.status = status

    def __init__(self, status, text=""):
        super().__init__(f"<HttpError {status} {text}>")
        self.resp = self._Resp(status)


@pytest.mark.parametrize(
    ("status", "text", "reason"),
    [
        (429, "", "rate_limit"),
        (403, "reason: userRateLimitExceeded", "rate_limit"),
        (403, "reason: rateLimitExceeded", "rate_limit"),
        (401, "", "auth"),
        (403, "reason: forbidden", None),
        (404, "", None),
        (500, "", None),
    ],
)
def test_which_drive_errors_cut_the_run(status, text, reason):
    assert gd_source._cut_reason(_HttpError(status, text)) == reason


def _service_failing_listing_after(pages_ok, status, text=""):
    """A fake whose files.list fails with ``status`` on the page after ``pages_ok`` pages."""
    files = [_file_meta(f"f{i:02d}") for i in range(6)]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})
    real_list = _FakeFilesResource.list
    calls = []

    def failing(self, *args, **kwargs):
        calls.append(1)
        if len(calls) > pages_ok:
            return _FakeRequest(error=_HttpError(status, text))
        return real_list(self, *args, **kwargs)

    return service, files, failing


def test_rate_limit_mid_run_keeps_progress_and_resumes(monkeypatch):
    files = [_file_meta(f"f{i}") for i in range(6)]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})
    fetched = []
    limited = {"on": True}

    def extract(_client, file_id, *_args):
        if limited["on"] and file_id == "f3":
            raise gd_source._CutShort("rate_limit")
        fetched.append(file_id)
        return "x"

    monkeypatch.setattr(gd_source, "extract_file_content", extract)
    state, stats = {"known_ids": ["gone"], "versions": {"gone": "v"}}, {}
    rows = list(gd_source._iter_rows(service, _config(), state, stats))

    assert [row["id"] for row in rows] == ["f0", "f1", "f2"]  # raised nothing; rows kept
    assert stats["failed"] == 1
    assert stats["failed_rate_limit"] == 1
    assert "page_token" not in state
    assert "gone" in state["known_ids"]  # an incomplete run reconciles nothing as absent
    assert state["versions"]["gone"] == "v"
    assert sorted(state["versions"]) == ["f0", "f1", "f2", "gone"]

    limited["on"] = False
    fetched.clear()
    rows = list(gd_source._iter_rows(service, _config(), state, stats))
    assert sorted(fetched) == ["f3", "f4", "f5"]  # only what the cut left undone
    assert stats["failed"] == 0
    assert state["page_token"] == "token-0"
    assert "gone" not in state["known_ids"]  # now the listing is complete


def test_429_on_a_download_cuts_the_run_instead_of_failing_every_file(monkeypatch):
    files = [_file_meta(f"f{i}") for i in range(4)]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})

    class _Files:
        def __init__(self):
            self.calls = []

        def export(self, fileId, mimeType):
            self.calls.append(fileId)
            raise _HttpError(429)

    resource = _Files()
    service.files = lambda: resource  # downloads only; listing is stubbed below
    monkeypatch.setattr(gd_source, "extract_file_content", _REAL_EXTRACT)
    monkeypatch.setattr(gd_source, "_list_files_in_scope", lambda *_a: iter(files))
    stats = {}
    rows = list(gd_source._iter_rows(service, _config(), {}, stats))
    assert rows == []
    assert stats["failed_rate_limit"] == 1
    assert "failed_content_extraction" not in stats
    assert resource.calls == ["f0"]  # it stopped at the first limited file


def test_expired_token_before_anything_is_raised_so_the_host_reconnects(monkeypatch):
    files = [_file_meta("f0")]
    service = FakeDriveService({"root": files}, {"f0": files[0]})
    monkeypatch.setattr(
        gd_source,
        "extract_file_content",
        lambda *_a: (_ for _ in ()).throw(gd_source._CutShort("auth")),
    )
    with pytest.raises(gd_source.DriveAuthError):
        list(gd_source._iter_rows(service, _config(), {}))


def test_expired_token_after_rows_is_a_cut_not_an_error(monkeypatch):
    files = [_file_meta(f"f{i}") for i in range(3)]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})

    def extract(_client, file_id, *_args):
        if file_id == "f2":
            raise gd_source._CutShort("auth")
        return "x"

    monkeypatch.setattr(gd_source, "extract_file_content", extract)
    stats = {}
    rows = list(gd_source._iter_rows(service, _config(), {}, stats))
    assert [row["id"] for row in rows] == ["f0", "f1"]
    assert stats["failed_auth"] == 1


def test_rate_limit_while_reading_the_changes_feed_keeps_the_token(monkeypatch):
    service = FakeDriveService({"root": []}, {})

    def limited(*_args, **_kwargs):
        raise gd_source._CutShort("rate_limit")

    monkeypatch.setattr(gd_source, "_list_changed_file_ids", limited)
    state, stats = {"page_token": "t0", "known_ids": ["a"], "versions": {"a": "v"}}, {}
    assert list(gd_source._iter_rows(service, _config(), state, stats)) == []
    assert state == {"page_token": "t0", "known_ids": ["a"], "versions": {"a": "v"}}
    assert stats["failed_rate_limit"] == 1


def test_rate_limit_during_changed_file_fetch_keeps_the_token_and_what_was_fetched(monkeypatch):
    files = [_file_meta(f"f{i}") for i in range(3)]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})
    service.changes_by_token["t0"] = {
        "changes": [{"fileId": f"f{i}", "removed": False} for i in range(3)],
        "newStartPageToken": "t1",
    }

    def extract(_client, file_id, *_args):
        if file_id == "f1":
            raise gd_source._CutShort("rate_limit")
        return "x"

    monkeypatch.setattr(gd_source, "extract_file_content", extract)
    state, stats = {"page_token": "t0"}, {}
    rows = list(gd_source._iter_rows(service, _config(), state, stats))
    assert [row["id"] for row in rows] == ["f0"]
    assert state["page_token"] == "t0"
    assert state["versions"] == {"f0": "2026-01-01T00:00:00Z"}
    assert stats["failed_rate_limit"] == 1


def test_listing_error_that_is_not_a_limit_still_raises(monkeypatch):
    service, _, failing = _service_failing_listing_after(0, 500)
    monkeypatch.setattr(_FakeFilesResource, "list", failing)
    with pytest.raises(RuntimeError, match="failed to list"):
        list(gd_source._iter_rows(service, _config(), {}))


def test_rate_limit_on_the_listing_is_a_cut(monkeypatch):
    service, _, failing = _service_failing_listing_after(0, 429)
    monkeypatch.setattr(_FakeFilesResource, "list", failing)
    stats = {}
    assert list(gd_source._iter_rows(service, _config(), {}, stats)) == []
    assert stats["failed_rate_limit"] == 1


def _ingested_then(service, file_meta):
    """Run a full scan with no size limit, so the file has a row and a version."""
    state = {}
    rows = list(gd_source._iter_rows(service, _config(max_file_size_mb=None), state))
    assert [row["id"] for row in rows] == [file_meta["id"]]
    assert state["versions"] == {file_meta["id"]: file_meta["modifiedTime"]}
    state.pop("page_token")
    return state


def test_lowered_limit_forgets_a_file_that_was_already_fetched(fake_content_extraction):
    big = dict(_file_meta("big"), size=str(30 * 1024 * 1024))
    service = FakeDriveService({"root": [big]}, {"big": big})
    state = _ingested_then(service, big)
    rows = list(gd_source._iter_rows(service, _config(max_file_size_mb=25), state))
    assert rows == [{"id": "big", "_deleted": True}]
    assert "big" not in state["versions"]


def test_lowered_limit_forgets_on_the_changes_feed_too(fake_content_extraction):
    big = dict(_file_meta("big"), size=str(30 * 1024 * 1024))
    service = FakeDriveService({"root": [big]}, {"big": big})
    state = _ingested_then(service, big)
    state["page_token"] = "t0"
    service.changes_by_token["t0"] = _feed({"fileId": "big", "removed": False})
    rows = list(gd_source._iter_rows(service, _config(max_file_size_mb=25), state))
    assert rows == [{"id": "big", "_deleted": True}]
    assert "big" not in state["versions"]


def _delta_for(service, file_id):
    service.changes_by_token["t0"] = _feed({"fileId": file_id, "removed": False})
    return {"page_token": "t0", "known_ids": [file_id], "versions": {file_id: "old"}}


def test_edited_file_that_now_reads_empty_is_forgotten(monkeypatch):
    doc = _file_meta("doc")
    service = FakeDriveService({"root": [doc]}, {"doc": doc})
    monkeypatch.setattr(gd_source, "extract_file_content", lambda *_a: "   ")
    state = _delta_for(service, "doc")
    stats = {}
    rows = list(gd_source._iter_rows(service, _config(), state, stats))
    assert rows == [{"id": "doc", "_deleted": True}]
    assert stats["skipped_empty_content"] == 1
    assert "doc" not in state["versions"]


def test_edited_file_that_became_an_unsupported_type_is_forgotten(fake_content_extraction):
    doc = _file_meta("doc", mime_type="image/png")
    service = FakeDriveService({"root": [doc]}, {"doc": doc})
    state = _delta_for(service, "doc")
    rows = list(gd_source._iter_rows(service, _config(), state))
    assert rows == [{"id": "doc", "_deleted": True}]
    assert "doc" not in state["versions"]


def test_edited_doc_over_googles_export_limit_is_forgotten_and_named(monkeypatch):
    doc = _file_meta("doc")
    service = FakeDriveService({"root": [doc]}, {"doc": doc})

    def too_large(*_args):
        raise gd_source.ExportTooLargeError("doc")

    monkeypatch.setattr(gd_source, "extract_file_content", too_large)
    state = _delta_for(service, "doc")
    stats = {}
    rows = list(gd_source._iter_rows(service, _config(), state, stats))
    assert rows == [{"id": "doc", "_deleted": True}]
    assert stats["failed_export_too_large"] == 1
    assert state["page_token"] == "t1"  # permanent: the cursor still moves


def test_missing_loader_keeps_what_an_earlier_run_ingested(monkeypatch):
    doc = _file_meta("doc")
    service = FakeDriveService({"root": [doc]}, {"doc": doc})

    def no_loader(*_args):
        raise gd_source.LoaderUnavailableError("no docling")

    monkeypatch.setattr(gd_source, "extract_file_content", no_loader)
    state = _delta_for(service, "doc")
    rows = list(gd_source._iter_rows(service, _config(), state, {}))
    assert rows == []  # an environment problem must not delete the file's memory
    assert state["versions"] == {}  # but nothing was fetched at this version either


def test_files_that_were_never_ingested_cause_no_tombstones(fake_content_extraction):
    files = [
        _file_meta("image", mime_type="image/png"),
        dict(_file_meta("huge"), size=str(30 * 1024 * 1024)),
    ]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})
    state, stats = {}, {}
    assert list(gd_source._iter_rows(service, _config(max_file_size_mb=25), state, stats)) == []
    assert stats["deleted"] == 0
    assert state["versions"] == {}


def test_a_failed_file_keeps_no_version_on_either_path(monkeypatch):
    doc = _file_meta("doc")
    service = FakeDriveService({"root": [doc]}, {"doc": doc})

    def no_loader(*_args):
        raise gd_source.LoaderUnavailableError("no docling")

    monkeypatch.setattr(gd_source, "extract_file_content", no_loader)
    state = {}
    list(gd_source._iter_rows(service, _config(), state))
    assert state["versions"] == {}
    delta = _delta_for(service, "doc")
    list(gd_source._iter_rows(service, _config(), delta))
    assert delta["versions"] == {}


@pytest.mark.parametrize(
    "unreadable",
    [drive_pdfs.locked_pdf(), *drive_pdfs.CORRUPT_PDFS.values()],
    ids=["password protected", *drive_pdfs.CORRUPT_PDFS],
)
def test_a_pdf_that_can_never_be_read_does_not_pin_the_cursor(monkeypatch, unreadable):
    """Issue 6235: two readable files and one unreadable PDF, synced three times."""
    locked = unreadable
    text = _file_meta("notes", mime_type="text/plain")
    pdf = _file_meta("locked", mime_type=PDF_MIME)
    service = FakeDriveService({"root": [text, pdf]}, {"notes": text, "locked": pdf})

    class _Request:
        def __init__(self, file_id):
            self.file_id = file_id

        def execute(self, num_retries=0):
            return locked if self.file_id == "locked" else b"hello"

    class _Files:
        def get_media(self, fileId, supportsAllDrives):
            return _Request(fileId)

        def list(self, *args, **kwargs):
            return _FakeFilesResource(service).list(*args, **kwargs)

        def get(self, *args, **kwargs):
            return _FakeFilesResource(service).get(*args, **kwargs)

    service.files = _Files
    downloads = []

    def counting(client, file_id, *args):
        downloads.append(file_id)
        return _REAL_EXTRACT(client, file_id, *args)

    monkeypatch.setattr(gd_source, "extract_file_content", counting)
    monkeypatch.setattr(
        gd_source, "_stream_media", lambda request, buffer: buffer.write(request.execute())
    )
    state, stats = {}, {}
    rows = list(gd_source._iter_rows(service, _config(), state, stats))
    assert [row["id"] for row in rows] == ["notes"]
    assert sorted(downloads) == ["locked", "notes"]
    assert state["page_token"] == "token-0"  # saved on the first run
    assert stats["failed"] == 1
    assert stats["failed_unreadable"] == 1
    assert "failed_content_extraction" not in stats

    service.changes_by_token["token-0"] = {"changes": [], "newStartPageToken": "token-1"}
    downloads.clear()
    assert list(gd_source._iter_rows(service, _config(), state, stats)) == []
    assert downloads == []  # incremental from here on: nothing is read again
    assert state["page_token"] == "token-1"


def test_file_edited_into_an_unreadable_one_is_forgotten_and_named(monkeypatch):
    doc = _file_meta("doc")
    service = FakeDriveService({"root": [doc]}, {"doc": doc})

    def unreadable(*_args):
        raise gd_source.UnreadableFileError("PdfReadError: broken")

    monkeypatch.setattr(gd_source, "extract_file_content", unreadable)
    state = _delta_for(service, "doc")
    stats = {}
    rows = list(gd_source._iter_rows(service, _config(), state, stats))
    assert rows == [{"id": "doc", "_deleted": True}]
    assert stats["failed_unreadable"] == 1
    assert state["page_token"] == "t1"
    assert "doc" not in state["versions"]


def test_unreadable_files_do_not_free_a_transient_failure(monkeypatch):
    files = [_file_meta("locked"), _file_meta("flaky")]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})

    def extract(_client, file_id, *_args):
        if file_id == "locked":
            raise gd_source.UnreadableFileError("locked")

    monkeypatch.setattr(gd_source, "extract_file_content", extract)
    state = {}
    list(gd_source._iter_rows(service, _config(), state, {}))
    assert "page_token" not in state


def test_files_list_pagination_lists_every_file_once(fake_content_extraction):
    files = [_file_meta(f"f{i}") for i in range(7)]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})
    service.page_size = 3
    rows = list(gd_source._iter_rows(service, _config(), {}))
    assert [row["id"] for row in rows] == [item["id"] for item in files]
    assert len([q for q in service.list_calls if "mimeType" not in q]) == 3  # three pages


def test_a_loader_failure_on_an_ingested_file_keeps_its_memory_and_the_cursor(monkeypatch):
    import sys
    from contextlib import contextmanager

    doc = _file_meta("doc", mime_type=DOCX_MIME)
    service = FakeDriveService({"root": [doc]}, {"doc": doc})

    @contextmanager
    def fake_download(_service, _file_id, _suffix):
        yield "/nonexistent/drive_file.docx"

    class _Engine:
        def get_loader(self, _path, _preferred):
            return object()

        async def load_file(self, _path, preferred_loaders=None, **_kwargs):
            raise RuntimeError("Document processing failed: HTTP 503 fetching a model")

    import cognee.infrastructure.loaders.get_loader_engine

    monkeypatch.setattr(
        sys.modules["cognee.infrastructure.loaders.get_loader_engine"],
        "get_loader_engine",
        lambda: _Engine(),
    )
    monkeypatch.setattr(gd_source, "extract_file_content", _REAL_EXTRACT)
    monkeypatch.setattr(gd_source, "_download_file", fake_download)
    state = _delta_for(service, "doc")
    stats = {}
    rows = list(gd_source._iter_rows(service, _config(), state, stats))
    assert rows == []  # no tombstone: the good document stays
    assert stats["failed_content_extraction"] == 1
    assert state["page_token"] == "t0"  # and the file is looked at again on the next run
    assert state["versions"] == {}


def test_the_size_bound_also_holds_on_the_changes_feed(monkeypatch):
    import threading
    import time

    monkeypatch.setattr(gd_source, "INFLIGHT_FETCH_BYTES", 100)
    files = [dict(_file_meta(f"f{i}"), size="60") for i in range(4)]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})
    service.changes_by_token["t0"] = _feed(
        *({"fileId": item["id"], "removed": False, "file": {"size": "60"}} for item in files)
    )
    running, peak, lock = [0], [0], threading.Lock()

    def extract(_client, _file_id, *_args):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.02)
        with lock:
            running[0] -= 1
        return "x"

    monkeypatch.setattr(gd_source, "extract_file_content", extract)
    state = {"page_token": "t0"}
    rows = list(
        gd_source._iter_rows(
            service,
            _config(max_workers=4, max_file_size_mb=None),
            state,
            service_factory=lambda: FakeDriveService(service.files_by_folder, service.file_by_id),
        )
    )
    assert len(rows) == 4
    assert peak[0] == 1  # 60 + 60 > 100: one at a time, as on a full scan


def test_the_feed_sizes_follow_the_last_change_of_each_file():
    service = FakeDriveService({"root": []}, {})
    service.changes_by_token["t0"] = _feed(
        {"fileId": "a", "removed": False, "file": {"size": "10"}},
        {"fileId": "a", "removed": False, "file": {"size": "30"}},
        {"fileId": "b", "removed": False, "file": {"trashed": False}},
    )
    sizes = {}
    changed, _, _ = gd_source._list_changed_file_ids(service, "t0", sizes=sizes)
    assert changed == {"a", "b"}
    assert sizes == {"a": 30, "b": 0}


def test_cut_run_does_not_resurrect_the_version_of_a_file_it_forgot(monkeypatch):
    files = [_file_meta("gone"), _file_meta("next")]
    service = FakeDriveService({"root": files}, {item["id"]: item for item in files})

    def extract(_client, file_id, *_args):
        if file_id == "next":
            raise gd_source._CutShort("rate_limit")
        return "   "  # "gone" was edited to blank: forgotten

    monkeypatch.setattr(gd_source, "extract_file_content", extract)
    state = {"known_ids": ["gone"], "versions": {"gone": "old"}}
    rows = list(gd_source._iter_rows(service, _config(), state, {}))
    assert rows == [{"id": "gone", "_deleted": True}]
    assert "gone" not in state["versions"]


def test_zero_workers_from_the_environment_is_an_error_not_the_default(monkeypatch):
    pytest.importorskip("dlt")
    monkeypatch.setenv("GOOGLE_DRIVE_MAX_WORKERS", "0")
    with pytest.raises(ValueError, match="GOOGLE_DRIVE_MAX_WORKERS"):
        gd_source.google_drive_source(folder_id="root", service=object())
    monkeypatch.setenv("GOOGLE_DRIVE_MAX_WORKERS", "3")
    gd_source.google_drive_source(folder_id="root", service=object())
