# Adapted from topoteretes/cognee-community (Apache-2.0), PR #370, commit 5e78950.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""google_calendar_source(): settings, credentials, sign-in builders and the dlt wiring.

The contract: every setting comes from ``GoogleCalendarSettings`` (the connector
reads no environment variable itself), missing credentials fail when the source
is built, and through a real dlt run a calendar that fails does not take the
finished calendars' state with it. No Google libraries and no network needed.
"""

import os
import stat
import sys
from datetime import datetime, timezone

import pytest
from fake_calendar import FakeCalendar, install_google_errors

from cognee.modules.integrations.google import GoogleAuthError, GoogleSettings
from cognee.tasks.ingestion.connectors import google_calendar as module
from cognee.tasks.ingestion.connectors.google_calendar import (
    GoogleCalendarSettings,
    build_calendar_service,
    google_calendar_source,
    list_calendars,
)
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

START = datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc)
TEAM = "team@group.calendar.google.com"

ENV = (
    "GOOGLE_CALENDAR_CLIENT_ID",
    "GOOGLE_CALENDAR_CLIENT_SECRET",
    "GOOGLE_CALENDAR_REFRESH_TOKEN",
    "GOOGLE_CALENDAR_CREDENTIALS_PATH",
    "GOOGLE_CALENDAR_TOKEN_PATH",
    "GOOGLE_CALENDAR_ACCESS_TOKEN",
    "GOOGLE_CALENDAR_IDS",
)


@pytest.fixture(autouse=True)
def no_calendar_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    install_google_errors(monkeypatch)


# ---------------------------------------------------------------------------
# Settings inherited from cognee
# ---------------------------------------------------------------------------
def test_settings_inherit_cognee_google_settings_and_read_their_prefix(monkeypatch):
    monkeypatch.setenv("GOOGLE_CALENDAR_CLIENT_ID", "client-123")
    monkeypatch.setenv("GOOGLE_CALENDAR_REFRESH_TOKEN", "refresh-abc")

    settings = GoogleCalendarSettings()

    assert isinstance(settings, GoogleSettings)
    assert settings.client_id == "client-123"
    assert settings.refresh_token == "refresh-abc"


def test_access_token_and_calendar_ids_are_settings_fields(monkeypatch):
    monkeypatch.setenv("GOOGLE_CALENDAR_ACCESS_TOKEN", "token-1")
    monkeypatch.setenv("GOOGLE_CALENDAR_IDS", "primary,team")

    settings = GoogleCalendarSettings()

    assert settings.access_token == "token-1"
    assert settings.ids == "primary,team"


def test_refresh_token_without_client_credentials_raises_cognee_auth_error():
    settings = GoogleCalendarSettings(refresh_token="refresh-abc")

    with pytest.raises(GoogleAuthError, match="missing_client_credentials"):
        module._build_service_from_refresh_token(settings)


# ---------------------------------------------------------------------------
# Installed-app flow
# ---------------------------------------------------------------------------
class _FakeCreds:
    valid = True
    expired = False
    refresh_token = "r"

    def __init__(self, calls=None):
        self.calls = calls if calls is not None else {}

    def to_json(self):
        return '{"token": "t"}'

    def refresh(self, request):
        self.calls["refreshed"] = True
        self.valid = True


class _ExpiredCreds(_FakeCreds):
    valid = False
    expired = True


class _ExpiredWithoutRefreshToken(_ExpiredCreds):
    refresh_token = None


def _fake_libraries(calls, cached=_FakeCreds):
    class Flow:
        @classmethod
        def from_client_secrets_file(cls, path, scopes):
            calls["scopes"] = scopes
            return cls()

        def run_local_server(self, port):
            calls["browser"] = True
            return _FakeCreds(calls)

    class Credentials:
        @staticmethod
        def from_authorized_user_file(path, scopes):
            return cached(calls)

    def build(name, version, credentials, cache_discovery):
        calls["built"] = (name, version)
        return "service"

    return lambda: (object, Credentials, Flow, build)


def test_first_login_writes_a_private_token_file_with_only_the_readonly_scope(
    tmp_path, monkeypatch
):
    calls: dict = {}
    monkeypatch.setattr(module, "_google_libraries", _fake_libraries(calls))
    secrets = tmp_path / "client_secret.json"
    secrets.write_text("{}")
    token = tmp_path / "token.json"

    service = build_calendar_service(str(secrets), str(token))

    assert service == "service"
    assert calls["built"] == ("calendar", "v3")
    assert calls["scopes"] == ["https://www.googleapis.com/auth/calendar.readonly"]
    assert calls["browser"] is True


# Windows has no POSIX file modes: os.stat reports 0o666 whatever the code asked for.
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")


@posix_only
def test_first_login_token_file_is_private(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_google_libraries", _fake_libraries({}))
    secrets = tmp_path / "client_secret.json"
    secrets.write_text("{}")
    token = tmp_path / "token.json"

    build_calendar_service(str(secrets), str(token))

    assert stat.S_IMODE(os.stat(token).st_mode) == 0o600


def test_cached_token_skips_the_browser(tmp_path, monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(module, "_google_libraries", _fake_libraries(calls))
    token = tmp_path / "token.json"
    token.write_text("{}")

    build_calendar_service(str(tmp_path / "missing.json"), str(token))

    assert "browser" not in calls


def test_expired_token_is_refreshed_and_written_back_without_the_browser(tmp_path, monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(module, "_google_libraries", _fake_libraries(calls, _ExpiredCreds))
    token = tmp_path / "token.json"
    token.write_text("{}")

    build_calendar_service(str(tmp_path / "missing.json"), str(token))

    assert calls["refreshed"] is True
    assert "browser" not in calls
    assert token.read_text() == '{"token": "t"}'


@posix_only
def test_a_loose_old_token_file_is_tightened_on_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_google_libraries", _fake_libraries({}, _ExpiredCreds))
    token = tmp_path / "token.json"
    token.write_text("{}")
    token.chmod(0o644)

    build_calendar_service(str(tmp_path / "missing.json"), str(token))

    assert stat.S_IMODE(os.stat(token).st_mode) == 0o600


def test_expired_token_without_a_refresh_token_asks_for_consent_again(tmp_path, monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        module, "_google_libraries", _fake_libraries(calls, _ExpiredWithoutRefreshToken)
    )
    secrets = tmp_path / "client_secret.json"
    secrets.write_text("{}")
    token = tmp_path / "token.json"
    token.write_text("{}")

    build_calendar_service(str(secrets), str(token))

    assert "refreshed" not in calls
    assert calls["browser"] is True


def test_missing_client_secrets_file_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_google_libraries", _fake_libraries({}))

    with pytest.raises(FileNotFoundError, match="GOOGLE_CALENDAR_CREDENTIALS_PATH"):
        build_calendar_service(str(tmp_path / "nope.json"), str(tmp_path / "token.json"))


# ---------------------------------------------------------------------------
# google_calendar_source()
# ---------------------------------------------------------------------------
def test_refuses_core_without_table_scoped_cleanup(monkeypatch):
    pytest.importorskip("dlt")
    monkeypatch.setattr(module.dlt_utils, "DOCUMENT_SYNC_VERSION", 0)

    with pytest.raises(RuntimeError, match="table-scoped"):
        google_calendar_source(service=FakeCalendar())


def test_resource_opts_into_document_mode_and_pipeline_scope():
    pytest.importorskip("dlt")
    resource = google_calendar_source(service=FakeCalendar(), resource_name="work_calendar")

    assert resource.name == "work_calendar"
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "google_calendar"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == "work_calendar"
    assert resource.cognee_sync_stats == {}


def test_missing_credentials_fail_when_the_source_is_built():
    pytest.importorskip("dlt")
    with pytest.raises(ValueError, match="credentials are missing"):
        google_calendar_source()


@pytest.mark.parametrize(
    "env",
    [
        {"GOOGLE_CALENDAR_ACCESS_TOKEN": "t"},
        {"GOOGLE_CALENDAR_REFRESH_TOKEN": "r"},
        {"GOOGLE_CALENDAR_CREDENTIALS_PATH": "/tmp/client.json"},
    ],
)
def test_any_one_auth_method_from_the_environment_is_enough(monkeypatch, env):
    pytest.importorskip("dlt")
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    google_calendar_source()  # builds without contacting Google


def test_access_token_argument_builds_the_hosted_client(monkeypatch):
    pytest.importorskip("dlt")
    tokens = []

    def build(token):
        tokens.append(token)
        return FakeCalendar()

    monkeypatch.setattr(module, "build_calendar_service_from_access_token", build)
    source = google_calendar_source(access_token="host-token")

    assert list(source) == []  # the empty primary calendar
    assert tokens == ["host-token"]


def test_empty_calendar_list_is_refused_instead_of_forgetting_everything():
    pytest.importorskip("dlt")
    with pytest.raises(ValueError, match="calendar_ids is empty"):
        google_calendar_source([], service=FakeCalendar())


def test_calendar_ids_come_from_settings_and_are_deduplicated():
    settings = GoogleCalendarSettings(ids=f" primary , {TEAM},primary")

    assert module._normalize_calendar_ids(None, settings) == ("primary", TEAM)
    assert module._normalize_calendar_ids("primary", settings) == ("primary",)
    assert module._normalize_calendar_ids(None, GoogleCalendarSettings()) == ("primary",)


def test_list_calendars_marks_the_primary_one():
    fake = FakeCalendar()
    fake.add_calendar(TEAM, "Team")

    calendars = list_calendars(fake)

    assert {"id": "lay@gmail.com", "name": "Lay Patel", "primary": True} in calendars
    assert {"id": TEAM, "name": "Team", "primary": False} in calendars


# ---------------------------------------------------------------------------
# Through a real dlt run: dlt rolls back the state of a raised run
# ---------------------------------------------------------------------------
def _pipeline(dlt, tmp_path):
    return dlt.pipeline(
        pipeline_name="gcal_state",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'gcal.db'}"),
        dataset_name="gcal_state",
        pipelines_dir=str(tmp_path / "pipelines"),
    )


def _ids(pipeline):
    with pipeline.sql_client() as client:
        return sorted(row[0] for row in client.execute_sql("SELECT id FROM google_calendar_events"))


def test_a_broken_calendar_does_not_cost_the_finished_calendars_their_state(tmp_path):
    # The reason the run does not raise on a partial failure: dlt throws the
    # state of a failed run away, so the finished calendar would be re-listed
    # from scratch and a deleted event would have no state to be forgotten from.
    dlt = pytest.importorskip("dlt")
    pipeline = _pipeline(dlt, tmp_path)
    fake = FakeCalendar()
    fake.add_calendar(TEAM, "Team")
    gone = fake.put_event("primary", "Gone", START)
    fake.put_event("primary", "Keep", START)
    fake.event_errors[TEAM] = 500
    calendars = ["primary", TEAM]

    source = google_calendar_source(calendars, service=fake)
    pipeline.run(source)

    assert source.cognee_sync_stats["failed"] == 1
    assert len(_ids(pipeline)) == 2

    fake.delete_event("primary", gone)
    fake.list_calls.clear()
    pipeline.run(google_calendar_source(calendars, service=fake))

    primary_calls = [c for c in fake.list_calls if c["calendarId"] == "lay@gmail.com"]
    assert primary_calls[0].get("syncToken"), "the primary calendar resumed from its saved token"
    assert len(_ids(pipeline)) == 1


def test_a_run_where_every_calendar_fails_raises_through_dlt_and_stores_nothing(tmp_path):
    dlt = pytest.importorskip("dlt")
    from dlt.pipeline.exceptions import PipelineStepFailed

    pipeline = _pipeline(dlt, tmp_path)
    fake = FakeCalendar()
    fake.put_event("primary", "Keep", START)
    fake.event_errors["lay@gmail.com"] = 503
    source = google_calendar_source(service=fake)

    with pytest.raises(PipelineStepFailed) as error:
        pipeline.run(source)

    assert "all 1 configured calendar(s)" in str(error.value.__cause__)
    assert source.cognee_sync_stats["failed"] == 1
    assert pipeline.state.get("sources") is None


def test_list_calendars_asks_for_hidden_calendars():
    # A hidden calendar missing from the list changes every row id of the primary
    # calendar and the name in a named calendar's content.
    fake = FakeCalendar()

    list_calendars(fake)

    assert [p["showHidden"] for p in fake.calendar_list_params] == [True]
