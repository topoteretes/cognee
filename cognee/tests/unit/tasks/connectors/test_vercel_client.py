# Adapted from topoteretes/cognee-community (Apache-2.0), PR #286, commit 0d9c0a7 (itssaharsh).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""The Vercel REST client: request shape, error mapping, retries and token hygiene.

Importing the connectors loads no HTTP client library, so ``requests`` is imported on the first
request. The token, the team id, a project or deployment id and the query string never reach an
error message, its ``__cause__`` or ``__context__``, a repr or a log line. The community source
leaked the URL through the exception chain into dlt's trace file; here the mapped error is
raised outside the ``except`` block, so it has no context. A 429 is never waited out.
"""

import logging
import subprocess
import sys
import textwrap

import pytest
import requests
from fake_vercel import Resp, sent_authorization

from cognee.tasks.ingestion.connectors.vercel import (
    BASE_URL,
    DEPLOYMENT_ROUTE,
    VercelAPIError,
    VercelAuthError,
    VercelClient,
    VercelNotFoundError,
    VercelRateLimitedError,
    VercelResponseError,
    build_vercel_service,
)

TOKEN = "SECRET-vercel-token-123"
SECRETS = (TOKEN, "team_acme", "dpl_SECRETUID", "teamId")

_IMPORT_WITHOUT_REQUESTS = textwrap.dedent(
    """
    import sys

    import cognee

    for name in list(sys.modules):
        if name.split(".")[0] == "requests" or name.startswith("cognee.tasks.ingestion.connectors"):
            del sys.modules[name]
    sys.modules["requests"] = None

    import cognee.tasks.ingestion.connectors.vercel as vercel

    assert not hasattr(vercel, "requests")
    """
)


def _client(monkeypatch, *responses):
    """A real client whose transport answers from ``responses`` in order."""
    calls, sleeps = [], []
    queue = list(responses)

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("requests.request", fake_request)
    return VercelClient(TOKEN, sleep=sleeps.append), calls, sleeps


def _get(client, **kwargs):
    return client.request(
        DEPLOYMENT_ROUTE,
        ids={"deployment": "dpl_SECRETUID"},
        params={"teamId": "team_acme"},
        **kwargs,
    )


def _chain(error: BaseException) -> list[BaseException]:
    seen = []
    while error is not None and error not in seen:
        seen.append(error)
        error = error.__cause__ or error.__context__
    return seen


def test_importing_the_connector_does_not_load_requests():
    subprocess.run([sys.executable, "-c", _IMPORT_WITHOUT_REQUESTS], check=True, timeout=120)


def test_a_request_sends_the_token_as_a_bearer_header_and_never_follows_a_redirect(monkeypatch):
    client, calls, _ = _client(monkeypatch, Resp(200, {"uid": "dpl_SECRETUID"}))

    assert _get(client) == {"uid": "dpl_SECRETUID"}

    method, url, kwargs = calls[0]
    assert (method, url) == ("GET", f"{BASE_URL}/v13/deployments/dpl_SECRETUID")
    assert sent_authorization(kwargs) == f"Bearer {TOKEN}"
    assert "Authorization" not in kwargs["headers"], "an auth object, so netrc cannot replace it"
    assert kwargs["params"] == {"teamId": "team_acme"}
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"]


def test_an_id_is_quoted_into_the_path(monkeypatch):
    client, calls, _ = _client(monkeypatch, Resp(200, {}))

    client.request(DEPLOYMENT_ROUTE, ids={"deployment": "../v2/user?x=1"})

    assert calls[0][1] == f"{BASE_URL}/v13/deployments/..%2Fv2%2Fuser%3Fx%3D1"


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, VercelAuthError),
        (403, VercelAuthError),
        (404, VercelNotFoundError),
        (410, VercelNotFoundError),
        (429, VercelRateLimitedError),
        (400, VercelAPIError),
        (302, VercelAPIError),
    ],
)
def test_error_statuses_map_to_typed_errors_that_name_the_route_only(monkeypatch, status, error):
    client, _, _ = _client(monkeypatch, Resp(status, {"error": {"message": f"{TOKEN} team_acme"}}))

    with pytest.raises(error, match=f"HTTP {status}") as raised:
        _get(client)

    assert type(raised.value) is error
    assert raised.value.status == status
    assert str(raised.value) == f"Vercel /v13/deployments/{{deployment}}: HTTP {status}"
    for link in _chain(raised.value):
        for secret in SECRETS:
            assert secret not in str(link)
            assert secret not in repr(link)
    assert raised.value.__context__ is None and raised.value.__cause__ is None


def test_a_rate_limit_is_never_retried(monkeypatch):
    client, calls, sleeps = _client(monkeypatch, Resp(429, {}))

    with pytest.raises(VercelRateLimitedError, match="HTTP 429"):
        _get(client)

    assert len(calls) == 1 and sleeps == []


def test_server_errors_and_network_failures_are_retried_with_a_short_backoff(monkeypatch):
    client, calls, sleeps = _client(
        monkeypatch, Resp(503, {}), requests.ConnectionError("boom"), Resp(200, {"ok": 1})
    )

    assert _get(client) == {"ok": 1}
    assert len(calls) == 3 and sleeps == [1, 2]


def test_a_network_error_that_persists_carries_no_url_and_no_context(monkeypatch):
    url_error = requests.ConnectionError(
        f"{BASE_URL}/v13/deployments/dpl_SECRETUID?teamId=team_acme"
    )
    client, calls, _ = _client(monkeypatch, url_error, url_error, url_error)

    with pytest.raises(VercelAPIError, match="no answer") as raised:
        _get(client)

    assert raised.value.status == 0
    assert str(raised.value) == "Vercel /v13/deployments/{deployment}: no answer"
    assert len(calls) == 3
    for link in _chain(raised.value):
        assert "team_acme" not in str(link) and "dpl_SECRETUID" not in str(link)


def test_a_server_error_that_persists_names_the_status_and_the_route_only(monkeypatch):
    client, _, sleeps = _client(monkeypatch, Resp(500, {}), Resp(502, {}), Resp(500, {}))

    with pytest.raises(VercelAPIError, match=r"^Vercel /v13/deployments/\{deployment\}: HTTP 500$"):
        _get(client)
    assert sleeps == [1, 2]


def test_check_active_runs_before_every_attempt_and_every_wait(monkeypatch):
    client, calls, _ = _client(monkeypatch, Resp(503, {}), Resp(503, {}), Resp(200, {}))
    checks = []

    _get(client, check_active=lambda: checks.append(len(calls)))

    # before the first attempt, then before each wait and each retry
    assert checks == [0, 1, 1, 2, 2]


def test_what_check_active_raises_stops_the_request_unwrapped(monkeypatch):
    client, calls, _ = _client(monkeypatch, Resp(200, {}))

    class Revoked(Exception):
        pass

    def revoked():
        raise Revoked

    with pytest.raises(Revoked):
        _get(client, check_active=revoked)
    assert calls == []


@pytest.mark.parametrize("payload", [ValueError("not json"), None])
def test_a_200_without_json_is_an_error_not_an_empty_answer(monkeypatch, payload):
    client, _, _ = _client(monkeypatch, Resp(200, payload))

    with pytest.raises(VercelResponseError, match="without a JSON body"):
        _get(client)


@pytest.mark.parametrize(
    "token",
    ["", "   ", "has space", "tab\there", "naïve", "line\nbreak", "nul\x00here", "del\x7fhere"],
)
def test_a_malformed_token_is_refused_without_echoing_it(token):
    with pytest.raises(
        ValueError, match="access token is empty or has an invalid format"
    ) as raised:
        VercelClient(token)
    assert token.strip() == "" or token not in str(raised.value)


def test_the_repr_hides_the_token():
    assert TOKEN not in repr(build_vercel_service(TOKEN))
    assert repr(build_vercel_service(TOKEN)) == "VercelClient(access_token=<redacted>)"


def test_no_log_line_carries_the_token_or_an_id(monkeypatch, caplog):
    client, _, _ = _client(monkeypatch, Resp(503, {}), Resp(503, {}), Resp(503, {}))

    with caplog.at_level(logging.DEBUG), pytest.raises(VercelAPIError, match="HTTP 503"):
        _get(client)

    for secret in SECRETS:
        assert secret not in caplog.text


def test_a_netrc_entry_for_the_host_never_replaces_the_bearer_token(monkeypatch, tmp_path):
    """requests reads ~/.netrc when no auth object is given and overwrites an Authorization
    header with it; the client passes an auth object, so the token is what goes out."""
    netrc = tmp_path / "netrc"
    netrc.write_text("machine api.vercel.com login someone password netrc-secret\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    sent = []

    def send(self, request, **kwargs):
        sent.append(request.headers.get("Authorization"))
        return Resp(200, {"uid": "dpl_SECRETUID"})

    monkeypatch.setattr(requests.sessions.Session, "send", send)

    VercelClient(TOKEN).request(DEPLOYMENT_ROUTE, ids={"deployment": "dpl_SECRETUID"})

    assert sent == [f"Bearer {TOKEN}"]


@pytest.mark.parametrize(
    "body",
    [
        '{"uid": "é"}'.encode("latin-1"),  # not UTF-8: requests would guess another charset
        b"\xff\xfe{}",
        b"[" * 100_000 + b"]" * 100_000,  # deeper than the JSON parser recurses
        b"null",
        b"",
    ],
    # Named ids: pytest puts the test id in PYTEST_CURRENT_TEST, and the 200 000 byte body as
    # an id is longer than Windows allows in an environment variable.
    ids=["latin-1", "utf-16-bom", "too-deep", "null", "empty"],
)
def test_a_200_that_is_not_utf8_json_is_a_response_error(monkeypatch, body):
    client, _, _ = _client(monkeypatch, Resp(200, body))

    with pytest.raises(VercelResponseError, match="without a JSON body"):
        _get(client)


def test_a_utf8_body_is_read_as_utf8(monkeypatch):
    client, _, _ = _client(monkeypatch, Resp(200, '{"name": "é"}'.encode()))

    assert _get(client) == {"name": "é"}
