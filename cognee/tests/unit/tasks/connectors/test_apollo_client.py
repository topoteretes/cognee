# Adapted from topoteretes/cognee-community (Apache-2.0), PR #304, commit 83a980e.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""The Apollo REST client: request shape, error mapping, retries and key hygiene.

Importing the connectors loads no HTTP client library, so ``requests`` is imported when
the first request is made. A key never reaches an error message, a repr or a log line, and
never leaves for another host: a redirect is an error. Apollo's rate-limit headers are read
off every answer that has them and kept per endpoint.
"""

import subprocess
import sys
import textwrap

import pytest
import requests
from requests.adapters import HTTPAdapter

from cognee.tasks.ingestion.connectors.apollo import (
    BASE_URL,
    ApolloAccessError,
    ApolloAPIError,
    ApolloAuthError,
    ApolloClient,
    ApolloNotFoundError,
    ApolloRateLimitedError,
    build_apollo_service,
)

KEY = "SECRET-apollo-key-123"
# The headers an answer for one endpoint carries, and the windows the client reads from them.
WINDOWS_HEADERS = {
    "x-minute-requests-left": "49",
    "x-hourly-requests-left": "399",
    "x-24-hour-requests-left": "1999",
}
WINDOWS = {"minute": 49, "hourly": 399, "daily": 1999}

# `import cognee` loads requests, so the script forgets it, and the connectors, first. A
# `None` entry makes any later `import requests` raise: the import below only succeeds
# when no connector module asks for it at import time.
_IMPORT_WITHOUT_REQUESTS = textwrap.dedent(
    """
    import sys

    import cognee

    for name in list(sys.modules):
        if name.split(".")[0] == "requests" or name.startswith("cognee.tasks.ingestion.connectors"):
            del sys.modules[name]
    sys.modules["requests"] = None

    import cognee.tasks.ingestion.connectors.apollo as apollo

    assert not hasattr(apollo, "requests")
    """
)


class _Response:
    def __init__(self, status: int, payload=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


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
    return ApolloClient(KEY, sleep=sleeps.append), calls, sleeps


def test_importing_the_connector_does_not_load_requests():
    subprocess.run([sys.executable, "-c", _IMPORT_WITHOUT_REQUESTS], check=True, timeout=120)


def test_a_request_sends_the_key_header_and_json_body(monkeypatch):
    client, calls, _ = _client(monkeypatch, _Response(200, {"contacts": []}, WINDOWS_HEADERS))

    assert client.request("POST", "/contacts/search", body={"page": 1}) == {"contacts": []}

    method, url, kwargs = calls[0]
    assert (method, url) == ("POST", "https://api.apollo.io/api/v1/contacts/search")
    assert kwargs["headers"]["x-api-key"] == KEY
    assert kwargs["json"] == {"page": 1}
    assert client.rate_limit == {"/contacts/search": WINDOWS}


@pytest.mark.parametrize(
    ("status", "payload", "error"),
    [
        (401, {}, ApolloAuthError),
        (403, {"error_code": "API_INACCESSIBLE"}, ApolloAccessError),
        (429, {}, ApolloRateLimitedError),
        (404, {"error": "Contact not found"}, ApolloNotFoundError),
        (422, {"error": "This contact has been deleted."}, ApolloNotFoundError),
        (422, {"error": "Oops! This contact does not exist in Apollo."}, ApolloNotFoundError),
        (422, {"error": "This account has been deleted!"}, ApolloNotFoundError),
        (422, {"error": "Page * per page number is over threshold."}, ApolloAPIError),
        # a record name in the message must not read as gone
        (422, {"error": "Account 'Not Found Logistics' could not be updated"}, ApolloAPIError),
        (422, {"error": "This contact does not exist in the stage you filtered"}, ApolloAPIError),
        (422, {"error": "This contact has been deleted from the sequence"}, ApolloAPIError),
        # Apollo documents no deleted-record code, so no code alone reads as gone. A known
        # limitation: when Apollo drops the legacy text on 2027-02-16 and answers a deleted
        # record with an error_details code only, these cases need to change with _is_gone.
        (422, {"error_details": {"code": "SEARCH.FILTER.STAGE_NOT_FOUND"}}, ApolloAPIError),
        (422, {"error_details": {"code": "CONTACTS.STATE.UNDELETED"}}, ApolloAPIError),
        (
            422,
            {
                "error": "This contact has been deleted.",
                "error_details": {"code": "REQUEST.VALIDATION.INVALID_PARAMETER"},
            },
            ApolloNotFoundError,
        ),
        (400, {}, ApolloAPIError),
    ],
)
def test_error_statuses_map_to_typed_errors(monkeypatch, status, payload, error):
    client, _, _ = _client(monkeypatch, _Response(status, payload))

    with pytest.raises(error) as raised:
        client.request("GET", "/contacts/c1")

    assert type(raised.value) is error
    assert KEY not in str(raised.value)


def test_a_403_names_the_endpoint_the_key_cannot_call(monkeypatch):
    client, _, _ = _client(monkeypatch, _Response(403))

    with pytest.raises(ApolloAccessError, match="/emailer_campaigns/search"):
        client.request("POST", "/emailer_campaigns/search", body={})


def test_a_403_on_a_record_never_names_its_id(monkeypatch):
    client, _, _ = _client(monkeypatch, _Response(403))

    with pytest.raises(ApolloAccessError) as raised:
        client.request("GET", "/contacts/6ac762b7d040e300200be4cf")

    assert "/contacts/{id}" in str(raised.value)
    assert "6ac762b7" not in str(raised.value)


@pytest.mark.parametrize("payload", [ValueError("not json"), "plain text"])
def test_a_200_without_a_json_object_is_an_error_not_an_empty_listing(monkeypatch, payload):
    client, _, _ = _client(monkeypatch, _Response(200, payload))

    with pytest.raises(ApolloAPIError, match="unreadable"):
        client.request("POST", "/contacts/search", body={})


def test_only_labels_may_answer_with_a_list(monkeypatch):
    client, _, _ = _client(
        monkeypatch, _Response(200, [{"id": "l1"}]), _Response(200, [{"id": "c1"}])
    )

    assert client.request("GET", "/labels") == [{"id": "l1"}]
    with pytest.raises(ApolloAPIError, match="unreadable"):
        client.request("POST", "/contacts/search", body={})


def test_server_errors_and_network_failures_are_retried(monkeypatch):
    client, calls, sleeps = _client(
        monkeypatch,
        _Response(503),
        requests.ConnectionError("boom"),
        _Response(200, {"ok": True}),
    )

    assert client.request("GET", "/contact_stages") == {"ok": True}
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_a_persistent_network_failure_raises_without_the_key(monkeypatch):
    client, _, _ = _client(monkeypatch, *[requests.ConnectionError(KEY)] * 3)

    with pytest.raises(ApolloAPIError) as raised:
        client.request("GET", "/contact_stages")

    assert KEY not in str(raised.value)
    assert raised.value.__cause__ is None


def test_the_key_is_never_shown():
    assert KEY not in repr(ApolloClient(KEY))


def test_the_service_a_host_builds_from_a_key_is_a_client():
    assert isinstance(build_apollo_service(KEY), ApolloClient)


@pytest.mark.parametrize("bad", ["", "   ", "has space", "tab\tkey", "ключ"])
def test_a_malformed_key_is_rejected_without_echoing_it(bad):
    with pytest.raises(ValueError) as raised:
        ApolloClient(bad)

    assert not bad.strip() or bad not in str(raised.value)


# --- rate limits are kept per endpoint, and read off an error answer that carries them ---


# The client reads the headers off every answer that carries them. That Apollo sends them on
# an error answer is not confirmed, so this pins only what the client does when it does.
@pytest.mark.parametrize(
    ("status", "payload", "error"),
    [
        (404, {}, ApolloNotFoundError),
        (422, {"error": "This contact has been deleted."}, ApolloNotFoundError),
        (403, {}, ApolloAccessError),
        (429, {}, ApolloRateLimitedError),
    ],
)
def test_an_error_answer_that_carries_the_windows_is_read_too(monkeypatch, status, payload, error):
    client, _, _ = _client(monkeypatch, _Response(status, payload, WINDOWS_HEADERS))

    with pytest.raises(error):
        client.request("GET", "/contacts/c1")

    assert client.rate_limit == {"/contacts/{id}": WINDOWS}


def test_the_windows_are_kept_per_endpoint(monkeypatch):
    other = {**WINDOWS_HEADERS, "x-minute-requests-left": "7"}
    client, _, _ = _client(
        monkeypatch,
        _Response(200, {"contacts": []}, WINDOWS_HEADERS),
        _Response(200, {"accounts": []}, other),
    )

    client.request("POST", "/contacts/search", body={})
    client.request("POST", "/accounts/search", body={})

    assert client.rate_limit == {
        "/contacts/search": WINDOWS,
        "/accounts/search": {**WINDOWS, "minute": 7},
    }


def test_an_answer_without_the_headers_keeps_what_is_known_about_its_endpoint(monkeypatch):
    client, _, _ = _client(
        monkeypatch,
        _Response(200, {"contacts": []}, WINDOWS_HEADERS),
        _Response(200, {"contacts": []}),
    )

    client.request("POST", "/contacts/search", body={})
    client.request("POST", "/contacts/search", body={})

    assert client.rate_limit == {"/contacts/search": WINDOWS}


# --- a redirect is an error and never carries the key to another host ---


def _hops(monkeypatch, *responses):
    """Answer at the HTTP adapter, below requests' own redirect handling, and record every hop."""
    sent, queue = [], list(responses)

    def send(self, request, **kwargs):
        sent.append((request.url, request.headers.get("x-api-key")))
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


# requests drops Authorization on a hop to another host but keeps any other header, so a
# followed redirect would hand the key to whoever the Location names.
@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_a_redirect_is_an_error_and_the_key_goes_nowhere_else(monkeypatch, status):
    sent = _hops(monkeypatch, (status, "https://collector.example.net/steal"), (200, None))

    with pytest.raises(ApolloAPIError) as raised:
        ApolloClient(KEY).request("GET", "/contacts/c1")

    assert sent == [(f"{BASE_URL}/contacts/c1", KEY)], "no second hop"
    assert str(raised.value) == (
        f"Apollo answered /contacts/{{id}} with a redirect (HTTP {status})"
    )
    assert "collector" not in str(raised.value)
