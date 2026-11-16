"""The Mixpanel REST client: request shape, error mapping, retries and credential hygiene.

Importing the connectors loads no HTTP client library, so ``requests`` is imported when
the first request is made. A credential never reaches an error message, a repr or a log
line, and never leaves for another host: a redirect is an error.
"""

import base64
import os
import subprocess
import sys
import textwrap
from datetime import date

import pytest
import requests
from requests.adapters import HTTPAdapter

from cognee_connector_mixpanel.source import (
    REGION_BASES,
    MixpanelAccessError,
    MixpanelAPIError,
    MixpanelAuthError,
    MixpanelClient,
    MixpanelRateLimitedError,
    build_mixpanel_service,
)

USERNAME = "svc-mixpanel.example"
SECRET = "SECRET-mixpanel-service-account-secret-123"
CREDENTIALS = f"Basic {base64.b64encode(f'{USERNAME}:{SECRET}'.encode()).decode()}"

# A `None` entry makes any later `import requests` raise: the import below only succeeds
# when no module of the package asks for it at import time (the package needs no host to
# import, so nothing loads requests before it).
_IMPORT_WITHOUT_REQUESTS = textwrap.dedent(
    """
    import sys

    sys.modules["requests"] = None

    import cognee_connector_mixpanel
    import cognee_connector_mixpanel.source as mixpanel

    assert not hasattr(mixpanel, "requests")
    """
)


class _Response:
    def __init__(self, status: int, payload=None, text: str | None = None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self._text = text if text is not None else "{}"
        self.headers = {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    @property
    def text(self):
        return self._text


def _client(monkeypatch, *responses):
    calls, sleeps = [], []
    queue = list(responses)

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("requests.request", fake_request)
    return MixpanelClient(USERNAME, SECRET, sleep=sleeps.append), calls, sleeps


def test_importing_the_connector_does_not_load_requests():
    # The package is found the way this process found it: installed, or on PYTHONPATH.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_REQUESTS], check=True, timeout=120, env=env
    )


def test_a_request_sends_basic_auth(monkeypatch):
    client, calls, _ = _client(monkeypatch, _Response(200, {"ok": True}))

    assert client.request("GET", "/cohorts/list") == {"ok": True}

    method, url, kwargs = calls[0]
    assert (method, url) == ("GET", "https://mixpanel.com/api/2.0/cohorts/list")
    assert kwargs["headers"]["Authorization"] == CREDENTIALS


@pytest.mark.parametrize("region", ["us", "eu"])
def test_each_region_targets_its_hosts(monkeypatch, region):
    client, calls, _ = _client(monkeypatch, _Response(200, {"ok": True}))

    MixpanelClient(USERNAME, SECRET, region=region, sleep=client._sleep).request(
        "GET", "/annotations"
    )
    _, url, _ = calls[0]
    assert url.startswith(REGION_BASES[region]["query"])


def test_an_unknown_region_is_rejected():
    with pytest.raises(ValueError, match="region"):
        MixpanelClient(USERNAME, SECRET, region="ap")


@pytest.mark.parametrize(
    ("status", "payload", "error"),
    [
        (401, {}, MixpanelAuthError),
        (403, {}, MixpanelAccessError),
        (429, {}, MixpanelRateLimitedError),
        (400, {"error": "bad request"}, MixpanelAPIError),
        (404, {}, MixpanelAPIError),
    ],
)
def test_error_statuses_map_to_typed_errors(monkeypatch, status, payload, error):
    client, _, _ = _client(monkeypatch, _Response(status, payload))

    with pytest.raises(error) as raised:
        client.request("GET", "/cohorts/list")

    assert type(raised.value) is error
    assert SECRET not in str(raised.value)
    assert USERNAME not in str(raised.value)


def test_a_403_names_the_endpoint_the_credential_cannot_call(monkeypatch):
    client, _, _ = _client(monkeypatch, _Response(403))

    with pytest.raises(MixpanelAccessError, match="/funnels/list"):
        client.request("GET", "/funnels/list", params={"project_id": "1"})


@pytest.mark.parametrize("payload", [ValueError("not json"), "plain text"])
def test_a_200_without_a_json_answer_is_an_error_not_an_empty_listing(monkeypatch, payload):
    client, _, _ = _client(monkeypatch, _Response(200, payload))

    with pytest.raises(MixpanelAPIError, match="unreadable"):
        client.request("GET", "/cohorts/list")


def test_a_200_json_array_answer_is_returned_as_is(monkeypatch):
    # Mixpanel's /annotations, /cohorts/list and /funnels/list answer with a bare array.
    client, _, _ = _client(monkeypatch, _Response(200, [{"id": 1}, {"id": 2}]))

    assert client.request("GET", "/cohorts/list") == [{"id": 1}, {"id": 2}]


def test_server_errors_and_network_failures_are_retried(monkeypatch):
    client, calls, sleeps = _client(
        monkeypatch,
        _Response(503),
        requests.ConnectionError("boom"),
        _Response(200, {"ok": True}),
    )

    assert client.request("GET", "/annotations") == {"ok": True}
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_a_revoked_connection_stops_a_retry_before_it_waits(monkeypatch):
    client, calls, sleeps = _client(monkeypatch, _Response(503), _Response(200, {"ok": True}))
    checks = []

    def check_active():
        checks.append(1)
        if len(checks) == 1:  # the first attempt runs unchecked; this is before its wait
            raise PermissionError("access revoked")

    with pytest.raises(PermissionError, match="access revoked"):
        client.request("GET", "/annotations", check_active=check_active)

    assert len(calls) == 1
    assert sleeps == []
    assert len(checks) == 1


def test_a_persistent_network_failure_raises_without_the_credentials(monkeypatch):
    client, _, _ = _client(monkeypatch, *[requests.ConnectionError(SECRET)] * 3)

    with pytest.raises(MixpanelAPIError) as raised:
        client.request("GET", "/annotations")

    assert SECRET not in str(raised.value)
    assert raised.value.__cause__ is None


def test_the_credentials_are_never_shown():
    client = MixpanelClient(USERNAME, SECRET)
    assert SECRET not in repr(client)
    assert USERNAME not in repr(client)


def test_the_service_a_host_builds_from_a_secret_is_a_client():
    service = build_mixpanel_service(
        service_account_username=USERNAME, service_account_secret=SECRET
    )
    assert isinstance(service, MixpanelClient)
    legacy = build_mixpanel_service(api_secret=SECRET)
    assert isinstance(legacy, MixpanelClient)


def test_the_two_credential_shapes_are_exclusive():
    with pytest.raises(ValueError, match="either"):
        build_mixpanel_service(
            service_account_username=USERNAME, service_account_secret=SECRET, api_secret=SECRET
        )
    with pytest.raises(ValueError, match="needs"):
        build_mixpanel_service()


@pytest.mark.parametrize("bad", ["has space", "tab\tsecret", "пароль"])
def test_a_malformed_secret_is_rejected_without_echoing_it(bad):
    with pytest.raises(ValueError) as raised:
        MixpanelClient(USERNAME, bad)

    assert bad not in str(raised.value)


@pytest.mark.parametrize("bad", ["", "   "])
def test_an_empty_username_is_rejected(bad):
    with pytest.raises(ValueError, match="username"):
        MixpanelClient(bad, SECRET)


def test_an_empty_password_is_the_legacy_api_secret_shape():
    # a legacy project API secret authenticates as the Basic auth username with an
    # empty password, so the client accepts it; the service account factory does not
    client = MixpanelClient(SECRET, "")
    assert SECRET not in repr(client)
    with pytest.raises(ValueError, match="needs"):
        build_mixpanel_service(service_account_username=USERNAME, service_account_secret="")


# --- the raw event export: newline-delimited JSON, retried like any request ---


def test_export_parses_newline_delimited_events(monkeypatch):
    lines = '\n{"event": "Signup", "properties": {"time": 1762800000}}\n\n{"event": "Login", "properties": {"time": 1762800001}}\n'
    client, calls, _ = _client(monkeypatch, _Response(200, text=lines))

    events = client.export_events(date(2026, 11, 10), date(2026, 11, 10))

    assert [event["event"] for event in events] == ["Signup", "Login"]
    _, url, kwargs = calls[0]
    assert url == "https://data.mixpanel.com/api/2.0/export"
    assert kwargs["params"] == {"from_date": "2026-11-10", "to_date": "2026-11-10"}


def test_an_unreadable_export_is_an_error(monkeypatch):
    client, _, _ = _client(monkeypatch, _Response(200, text="{not json"))

    with pytest.raises(MixpanelAPIError, match="unreadable"):
        client.export_events(date(2026, 11, 10), date(2026, 11, 10))


# --- a redirect is an error and never carries the credentials to another host ---


def _hops(monkeypatch, *responses):
    """Answer at the HTTP adapter, below requests' own redirect handling, and record every hop."""
    sent, queue = [], list(responses)

    def send(self, request, **kwargs):
        sent.append((request.url, request.headers.get("Authorization")))
        status, location = queue.pop(0)
        response = requests.Response()
        response.status_code = status
        response.url = request.url
        response.request = request
        response._content = b""
        if location:
            response.headers["Location"] = location
        return response

    monkeypatch.setattr(HTTPAdapter, "send", send)
    return sent


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_a_redirect_is_an_error_and_the_credentials_go_nowhere_else(monkeypatch, status):
    sent = _hops(monkeypatch, (status, "https://collector.example.net/steal"), (200, None))

    with pytest.raises(MixpanelAPIError) as raised:
        MixpanelClient(USERNAME, SECRET).request("GET", "/cohorts/list")

    assert sent == [("https://mixpanel.com/api/2.0/cohorts/list", CREDENTIALS)], "no second hop"
    assert "collector" not in str(raised.value)
    assert SECRET not in str(raised.value)
