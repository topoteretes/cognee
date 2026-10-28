"""Exercise the SDK's host-supplied credentials through Google's real transport."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("googleapiclient")

import httplib2
from google.auth import _helpers
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.errors import HttpError

from cognee.modules.integrations import ingestion
from cognee.tasks.ingestion.connectors import google_drive as drive


@pytest.fixture
def google_client(monkeypatch):
    clock = SimpleNamespace(
        now=datetime(2026, 10, 28, 12, tzinfo=timezone.utc).replace(tzinfo=None)
    )
    monkeypatch.setattr(_helpers, "utcnow", lambda: clock.now)
    requests = []

    class Transport:
        def request(self, uri, method="GET", body=None, headers=None, **kwargs):
            requests.append(headers["authorization"])
            return httplib2.Response({"status": "200"}), b"document text"

    class Service:
        def __init__(self, credentials):
            self.http = AuthorizedHttp(credentials, http=Transport())

        def files(self):
            return self

        def export(self, **kwargs):
            def execute(**kwargs):
                response, content = self.http.request("https://drive.test")
                if int(response.status) >= 400:  # what googleapiclient does with an error status
                    raise HttpError(response, content)
                return content

            return SimpleNamespace(execute=execute)

    monkeypatch.setattr(
        "googleapiclient.discovery.build",
        lambda api, version, credentials, **kwargs: Service(credentials),
    )
    return clock, requests


def _expiry(clock):
    return (clock.now + timedelta(hours=1)).replace(tzinfo=timezone.utc)


def _files():
    return [
        {
            "id": f"f{index}",
            "name": f"Document {index}",
            "mimeType": drive.GOOGLE_DOC_MIME_TYPE,
            "modifiedTime": "v1",
        }
        for index in range(3)
    ]


def _config():
    return drive._DriveConfig(
        folder_id="folder",
        auth_mode="oauth",
        credentials_path=None,
        token_path=None,
        include_subfolders=True,
    )


def test_one_folder_continues_after_token_expiry_without_losing_rows_or_cursor(
    monkeypatch, google_client
):
    clock, requests = google_client
    renew = Mock(side_effect=lambda used: ("new-token", _expiry(clock)))
    service = ingestion.build_service(
        "google_drive", "old-token", expires_at=_expiry(clock), refresh_access_token=renew
    )
    # Client construction must not invoke the host or make an API request.
    renew.assert_not_called()
    assert requests == []

    def listing(*args):
        for index, metadata in enumerate(_files()):
            if index == 1:
                clock.now += timedelta(hours=2)
            yield metadata

    monkeypatch.setattr(drive, "_list_files_in_scope", listing)
    monkeypatch.setattr(drive, "_get_start_page_token", lambda *args: "cursor-before-listing")
    state, stats = {}, {}
    rows = list(drive._iter_rows(service, _config(), state, stats))

    assert [row["id"] for row in rows] == ["f0", "f1", "f2"]
    assert requests == ["Bearer old-token", "Bearer new-token", "Bearer new-token"]
    renew.assert_called_once_with("old-token")
    assert state["page_token"] == "cursor-before-listing"
    assert state["versions"] == {"f0": "v1", "f1": "v1", "f2": "v1"}
    assert stats["failed"] == 0


@pytest.mark.parametrize("renewable", [False, True])
def test_refresh_failure_is_an_auth_cut_and_keeps_partial_progress(
    monkeypatch, google_client, renewable
):
    clock, requests = google_client
    renew = Mock(side_effect=drive.TokenRenewalError("renewal refused"))
    service = ingestion.build_service(
        "google_drive",
        "old-token",
        expires_at=_expiry(clock),
        refresh_access_token=renew if renewable else None,
    )

    def listing(*args):
        yield _files()[0]
        clock.now += timedelta(hours=2)
        yield _files()[1]

    monkeypatch.setattr(drive, "_list_files_in_scope", listing)
    monkeypatch.setattr(drive, "_get_start_page_token", lambda *args: "not-committed")
    state, stats = {}, {}
    rows = list(drive._iter_rows(service, _config(), state, stats))

    assert [row["id"] for row in rows] == ["f0"]
    assert state["versions"] == {"f0": "v1"}
    assert "page_token" not in state
    assert stats["failed_auth"] == 1
    assert "failed_content_extraction" not in stats
    assert requests == ["Bearer old-token"]
    if not renewable:
        renew.assert_not_called()


def test_a_renewal_callback_without_an_expiry_is_refused_at_build_time():
    with pytest.raises(ValueError, match="expires_at"):
        drive.build_drive_service_from_access_token(
            "old-token", refresh_access_token=Mock(return_value=("new-token", None))
        )


def test_an_expired_token_before_the_first_row_is_a_typed_auth_error(monkeypatch, google_client):
    clock, _ = google_client
    service = ingestion.build_service(
        "google_drive", "old-token", expires_at=clock.now - timedelta(seconds=1)
    )
    monkeypatch.setattr(drive, "_list_files_in_scope", lambda *args: iter(_files()))
    monkeypatch.setattr(drive, "_get_start_page_token", lambda *args: "cursor")
    with pytest.raises(drive.DriveAuthError):
        list(drive._iter_rows(service, _config(), {}))


def test_a_401_on_a_token_that_looks_valid_uses_the_host_refresh_handler(
    monkeypatch, google_client
):
    clock, requests = google_client
    renew = Mock(side_effect=lambda used: ("new-token", _expiry(clock)))
    service = drive.build_drive_service_from_access_token(
        "old-token", expires_at=_expiry(clock), refresh_access_token=renew
    )

    def rejected_once(uri, method="GET", body=None, headers=None, **kwargs):
        requests.append(headers["authorization"])
        status = "401" if len(requests) == 1 else "200"
        return httplib2.Response({"status": status}), b"document text"

    service.http.http.request = rejected_once
    assert drive.extract_file_content(service, "f", drive.GOOGLE_DOC_MIME_TYPE, "doc") == (
        "document text"
    )
    assert requests == ["Bearer old-token", "Bearer new-token"]
    renew.assert_called_once_with("old-token")


def test_a_host_that_returns_the_same_token_after_a_401_ends_the_run_as_auth(
    monkeypatch, google_client
):
    clock, requests = google_client
    renew = Mock(side_effect=lambda used: ("old-token", _expiry(clock)))
    service = drive.build_drive_service_from_access_token(
        "old-token", expires_at=_expiry(clock), refresh_access_token=renew
    )

    def always_rejected(uri, method="GET", body=None, headers=None, **kwargs):
        requests.append(headers["authorization"])
        return httplib2.Response({"status": "401"}), b""

    service.http.http.request = always_rejected
    monkeypatch.setattr(drive, "_list_files_in_scope", lambda *args: iter(_files()))
    monkeypatch.setattr(drive, "_get_start_page_token", lambda *args: "cursor")
    state, stats = {}, {}
    with pytest.raises(drive.DriveAuthError):
        list(drive._iter_rows(service, _config(), state, stats))
    assert set(requests) == {"Bearer old-token"}


def test_401_without_refresh_material_stops_as_auth_instead_of_skipping_files(
    monkeypatch, google_client
):
    _, requests = google_client
    service = ingestion.build_service("google_drive", "old-token")

    def rejected_after_first_file(uri, method="GET", body=None, headers=None, **kwargs):
        requests.append(headers["authorization"])
        status = "200" if len(requests) == 1 else "401"
        return httplib2.Response({"status": status}), b"document text"

    service.http.http.request = rejected_after_first_file
    monkeypatch.setattr(drive, "_list_files_in_scope", lambda *args: iter(_files()))
    monkeypatch.setattr(drive, "_get_start_page_token", lambda *args: "cursor")
    state, stats = {}, {}
    rows = list(drive._iter_rows(service, _config(), state, stats))
    assert [row["id"] for row in rows] == ["f0"]
    assert len(requests) == 2
    assert stats["failed_auth"] == 1
    assert "failed_content_extraction" not in stats
    assert state["versions"] == {"f0": "v1"}
    assert "page_token" not in state


def test_auth_cut_and_resume_preserve_rows_through_a_real_dlt_pipeline(
    monkeypatch, google_client, tmp_path
):
    import dlt

    clock, requests = google_client
    pipeline = dlt.pipeline(
        pipeline_name="drive_auth_resume",
        dataset_name="drive",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'drive.db'}"),
        pipelines_dir=str(tmp_path / "pipelines"),
    )

    def interrupted_listing(*args):
        yield _files()[0]
        clock.now += timedelta(hours=2)
        yield from _files()[1:]

    start = Mock(return_value="start-cursor")
    monkeypatch.setattr(drive, "_get_start_page_token", start)
    monkeypatch.setattr(drive, "_list_files_in_scope", interrupted_listing)
    first = drive.google_drive_source(
        folder_id="folder",
        resource_name="folder_rows",
        service=ingestion.build_service("google_drive", "old-token", expires_at=_expiry(clock)),
    )
    pipeline.run(first)
    assert first.cognee_sync_stats["failed_auth"] == 1
    with pipeline.sql_client() as sql:
        assert sql.execute_sql("SELECT id FROM folder_rows") == [("f0",)]

    monkeypatch.setattr(drive, "_list_files_in_scope", lambda *args: iter(_files()))

    def resumed_source():
        return drive.google_drive_source(
            folder_id="folder",
            resource_name="folder_rows",
            service=ingestion.build_service(
                "google_drive", "resumed-token", expires_at=_expiry(clock)
            ),
        )

    second = resumed_source()
    pipeline.run(second)
    assert second.cognee_sync_stats["unchanged"] == 1
    assert second.cognee_sync_stats["failed"] == 0
    assert requests == ["Bearer old-token", "Bearer resumed-token", "Bearer resumed-token"]
    with pipeline.sql_client() as sql:
        assert sql.execute_sql("SELECT id FROM folder_rows ORDER BY id") == [
            ("f0",),
            ("f1",),
            ("f2",),
        ]

    # The interrupted listing did not advance the cursor; the completed retry did.
    assert start.call_count == 2
    changes = Mock(return_value=(set(), set(), "next-cursor"))
    monkeypatch.setattr(drive, "_list_changed_file_ids", changes)
    pipeline.run(resumed_source())
    assert start.call_count == 2
    assert changes.call_args.args[1] == "start-cursor"
    assert len(requests) == 3
