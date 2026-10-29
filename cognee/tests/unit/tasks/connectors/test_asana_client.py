# Adapted from topoteretes/cognee-community (Apache-2.0), PR #349, commit ef32699 (AbdulWasih05).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""The Asana REST client: request shape, error mapping, retries and token hygiene.

Importing the connectors loads no HTTP client library, so ``requests`` is imported when the
first request is made. The token never reaches an error message, a ``__cause__``, a repr or a log
line, and a rate limit is never waited out: a 429 is an error at once, whatever ``Retry-After``
says, because the walk holds the process-wide staging lock.
"""

import logging
import subprocess
import sys
import textwrap

import pytest
import requests
from fake_asana import FakeAsana, Resp

from cognee.tasks.ingestion.connectors.asana import (
    BASE_URL,
    AsanaAPIError,
    AsanaAuthError,
    AsanaClient,
    AsanaRateLimitedError,
    AsanaResponseError,
    AsanaSyncTokenExpiredError,
    build_asana_service,
)

TOKEN = "SECRET-asana-token-123"

# `import cognee` loads requests, so the script forgets it, and the connectors, first. A `None`
# entry makes any later `import requests` raise: the import below only succeeds when no connector
# module asks for it at import time.
_IMPORT_WITHOUT_REQUESTS = textwrap.dedent(
    """
    import sys

    import cognee

    for name in list(sys.modules):
        if name.split(".")[0] == "requests" or name.startswith("cognee.tasks.ingestion.connectors"):
            del sys.modules[name]
    sys.modules["requests"] = None

    import cognee.tasks.ingestion.connectors.asana as asana

    assert not hasattr(asana, "requests")
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
    return AsanaClient(TOKEN, sleep=sleeps.append), calls, sleeps


def test_importing_the_connector_does_not_load_requests():
    subprocess.run([sys.executable, "-c", _IMPORT_WITHOUT_REQUESTS], check=True, timeout=120)


def test_a_request_sends_the_bearer_token_and_the_params(monkeypatch):
    client, calls, _ = _client(monkeypatch, Resp(200, {"data": []}))

    assert client.request("GET", "/tasks", params={"project": "1"}) == {"data": []}

    method, url, kwargs = calls[0]
    assert (method, url) == ("GET", f"{BASE_URL}/tasks")
    assert kwargs["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert kwargs["params"] == {"project": "1"}
    assert kwargs["timeout"]


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, AsanaAuthError),
        (429, AsanaRateLimitedError),
        (403, AsanaAPIError),
        (404, AsanaAPIError),
        (400, AsanaAPIError),
        (412, AsanaAPIError),  # no sync token in the body: not the events answer
    ],
)
def test_error_statuses_map_to_typed_errors(monkeypatch, status, error):
    client, _, _ = _client(monkeypatch, Resp(status, {"errors": [{"message": "No way"}]}))

    with pytest.raises(error) as raised:
        client.request("GET", "/tasks/1234567890")

    assert type(raised.value) is error
    assert raised.value.status == status
    assert str(raised.value) == f"Asana /tasks/{{gid}}: HTTP {status}"


def test_the_error_message_never_names_a_gid_and_asanas_explanation_is_an_attribute(monkeypatch):
    client, _, _ = _client(
        monkeypatch, Resp(403, {"errors": [{"message": "Task 1234567890 is private"}]})
    )

    with pytest.raises(AsanaAPIError) as raised:
        client.request("GET", "/tasks/1234567890/stories")

    assert str(raised.value) == "Asana /tasks/{gid}/stories: HTTP 403"
    assert raised.value.endpoint == "/tasks/{gid}/stories"
    assert raised.value.detail == "Task 1234567890 is private"


def test_a_412_with_a_sync_token_is_the_expired_token_answer(monkeypatch):
    client, _, _ = _client(monkeypatch, Resp(412, {"errors": [], "sync": "fresh-token"}))

    with pytest.raises(AsanaSyncTokenExpiredError) as raised:
        client.request("GET", "/events", params={"resource": "1"})

    assert raised.value.sync_token == "fresh-token"


@pytest.mark.parametrize("payload", [ValueError("not json"), "plain text", ["a list"], None])
def test_a_200_without_a_json_object_is_an_error_not_an_empty_answer(monkeypatch, payload):
    client, _, _ = _client(monkeypatch, Resp(200, payload))

    with pytest.raises(AsanaResponseError, match="without a JSON object"):
        client.request("GET", "/tasks")


def test_server_errors_and_network_failures_are_retried_with_a_short_backoff(monkeypatch):
    client, calls, sleeps = _client(
        monkeypatch, Resp(503, {}), requests.ConnectionError("boom"), Resp(200, {"data": {}})
    )

    assert client.request("GET", "/projects/1") == {"data": {}}
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_a_persistent_server_error_gives_up_after_the_retries(monkeypatch):
    client, calls, sleeps = _client(monkeypatch, *[Resp(500, {})] * 3)

    with pytest.raises(AsanaAPIError) as raised:
        client.request("GET", "/projects/1")

    assert raised.value.status == 500
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_a_persistent_network_failure_raises_without_the_token(monkeypatch):
    client, _, _ = _client(monkeypatch, *[requests.ConnectionError(TOKEN)] * 3)

    with pytest.raises(AsanaAPIError) as raised:
        client.request("GET", "/projects/1")

    assert raised.value.status == 0
    assert str(raised.value) == "Asana /projects/{gid}: no answer"
    assert TOKEN not in str(raised.value) and TOKEN not in repr(raised.value.args)
    # dlt's trace file reads __context__ too: nothing may hold the transport's own error
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def _traced(monkeypatch, *responses):
    """A real client, and one list with every check, send and sleep in the order they happen."""
    events: list[str] = []
    queue = list(responses)

    def fake_request(method, url, **kwargs):
        events.append("send")
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("requests.request", fake_request)
    return AsanaClient(TOKEN, sleep=lambda seconds: events.append(f"sleep {seconds}")), events


def test_the_host_is_asked_before_each_attempt_and_before_each_wait(monkeypatch):
    """The check comes before the sleep, so a revoked run does not wait out the backoff first."""
    client, events = _traced(monkeypatch, *[Resp(503, {})] * 3)

    with pytest.raises(AsanaAPIError):
        client.request("GET", "/projects/1", check_active=lambda: events.append("check"))

    assert events == ["check", "send", "check", "sleep 1", "send", "check", "sleep 2", "send"]


def test_a_retried_request_asks_the_host_once_per_attempt(monkeypatch):
    client, events = _traced(monkeypatch, Resp(500, {}), Resp(200, {"data": {}}))

    client.request("GET", "/projects/1", check_active=lambda: events.append("check"))

    assert events == ["check", "send", "check", "sleep 1", "send"]


@pytest.mark.parametrize(
    ("revoked_at", "sent", "slept"),
    [(1, 0, []), (2, 1, []), (3, 2, [1])],
)
def test_a_revoked_host_stops_the_retries_before_the_attempt_and_the_wait(
    monkeypatch, revoked_at, sent, slept
):
    """The attempt the host refused is never sent, and no wait follows the refusal."""
    client, events = _traced(monkeypatch, *[Resp(503, {})] * 3)
    asked = []

    def check():
        asked.append(1)
        if len(asked) == revoked_at:
            raise PermissionError("access revoked")

    with pytest.raises(PermissionError, match="access revoked"):
        client.request("GET", "/projects/1", check_active=check)

    assert len(asked) == revoked_at
    assert events.count("send") == sent
    assert [event for event in events if event.startswith("sleep")] == [
        f"sleep {seconds}" for seconds in slept
    ]


def test_what_the_host_raises_is_neither_wrapped_nor_retried(monkeypatch):
    """A host bridge can fail with a requests error: it must not be read as a network failure."""
    client, events = _traced(monkeypatch, Resp(503, {}), Resp(200, {"data": {}}))
    error = requests.ConnectionError("host bridge down")
    asked = []

    def bridge_down():
        asked.append(1)
        if len(asked) == 2:
            raise error

    with pytest.raises(requests.ConnectionError) as raised:
        client.request("GET", "/projects/1", check_active=bridge_down)

    assert raised.value is error and error.__cause__ is None
    assert len(asked) == 2 and events == ["send"]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_a_client_error_is_not_retried(monkeypatch, status):
    client, calls, sleeps = _client(monkeypatch, Resp(status, {}), Resp(200, {}))

    with pytest.raises(AsanaAPIError):
        client.request("GET", "/projects/1")

    assert len(calls) == 1 and sleeps == []


# Whatever Retry-After says (negative, not a number, infinite, huge, an HTTP date), a 429 is
# an error at once and nothing sleeps: the old client parsed it into a wait, and a negative or
# NaN value crashed the retry loop.
@pytest.mark.parametrize(
    "retry_after",
    ["-5", "nan", "-inf", "inf", "1e308", "0", "30", "soon", "Wed, 21 Oct 2026 07:28:00 GMT", ""],
)
def test_a_429_is_never_waited_out(monkeypatch, retry_after):
    headers = {"Retry-After": retry_after}
    client, calls, sleeps = _client(monkeypatch, Resp(429, {}, headers), Resp(200, {}))

    with pytest.raises(AsanaRateLimitedError):
        client.request("GET", "/tasks")

    assert len(calls) == 1
    assert sleeps == []


# --- the token ---


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "SEC RET",
        "SEC\tRET",
        "SECRET\x00",
        "SECRET\x7f",
        "SECR\u00c9T",
        "\u043a\u043b\u044e\u0447",
        "SEC\u2028RET",
        None,
        123,
    ],
)
def test_a_malformed_token_is_refused_without_echoing_it(bad):
    with pytest.raises(ValueError) as raised:
        AsanaClient(bad)

    assert str(raised.value) == "The Asana access token is empty or has an invalid format"
    assert not isinstance(bad, str) or not bad.strip() or bad.strip() not in str(raised.value)


def test_the_token_is_never_shown():
    assert TOKEN not in repr(AsanaClient(TOKEN))
    assert TOKEN not in repr(build_asana_service(TOKEN)) and isinstance(
        build_asana_service(TOKEN), AsanaClient
    )


@pytest.mark.parametrize("status", [401, 403, 404, 412, 429, 500, 502])
def test_the_token_is_in_no_error_cause_log_or_detail_even_when_asana_echoes_it(
    monkeypatch, caplog, status
):
    echo = {"errors": [{"message": f"Bad header: Bearer {TOKEN}"}], "sync": "s"}
    client, _, _ = _client(monkeypatch, *[Resp(status, echo)] * 3)

    with caplog.at_level(logging.DEBUG), pytest.raises(AsanaAPIError) as raised:
        client.request("GET", "/tasks/1")

    error = raised.value
    assert TOKEN not in str(error) and TOKEN not in repr(error) and TOKEN not in repr(error.args)
    assert error.__cause__ is None and error.__context__ is None
    assert TOKEN not in error.detail
    assert TOKEN not in caplog.text


def test_the_token_is_cut_out_before_the_text_is_shortened(monkeypatch):
    """The token that straddles the 200 character cut must not leave a prefix behind."""
    message = "x" * 195 + TOKEN
    client, _, _ = _client(monkeypatch, Resp(403, {"errors": [{"message": message}]}))

    with pytest.raises(AsanaAPIError) as raised:
        client.request("GET", "/tasks/1")

    assert len(raised.value.detail) <= 200
    assert TOKEN[:4] not in raised.value.detail


@pytest.mark.parametrize(
    "body", [{}, {"errors": []}, {"errors": ["x"]}, {"errors": [{"message": 5}]}]
)
def test_an_answer_without_a_message_has_no_detail(monkeypatch, body):
    client, _, _ = _client(monkeypatch, Resp(403, body))

    with pytest.raises(AsanaAPIError) as raised:
        client.request("GET", "/tasks/1")

    assert raised.value.detail == ""


def test_the_fake_over_http_reaches_the_client_the_way_a_host_sees_it(monkeypatch):
    from fake_asana import serve_over_http

    fake = FakeAsana()
    fake.add_project("P1")
    fake.add_task("1")
    sent: list[str] = []
    monkeypatch.setattr("requests.request", serve_over_http(fake, authorizations=sent))
    client = AsanaClient(TOKEN)

    body = client.request("GET", "/tasks", params={"project": "P1", "opt_fields": "modified_at"})

    assert [item["gid"] for item in body["data"]] == ["1"]
    assert sent == [f"Bearer {TOKEN}"]
