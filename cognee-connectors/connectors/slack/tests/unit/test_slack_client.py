"""``SlackClient`` through httpx with a fake transport: errors, retries, and what never leaks.

Copied from cognee.tests.unit.tasks.connectors.test_slack_client (cognee dev, 2026-11-12).

The token, the channel id and the request URL must not appear in any exception, and the
token and channel id in no log line, whatever Slack answers. These tests go through the
real client (``httpx.MockTransport``), not a prebuilt service, because the client is the
code that builds the messages.
"""

import logging
from urllib.parse import parse_qs

import httpx
import pytest

from cognee_connector_slack.source import (
    API_URL,
    SlackAPIError,
    SlackAuthError,
    SlackClient,
    SlackMissingScopeError,
    SlackRateLimitedError,
    SlackUnavailableError,
    _iter_rows,
    build_slack_service,
    slack_source,
)

TOKEN = "xoxb-1111-2222-SECRETSECRET"
CHANNEL = "C0SECRETCHAN"
TEAM = "T0TEAM"


def _client(handler, sleeps: list | None = None) -> SlackClient:
    sleeps = sleeps if sleeps is not None else []
    return SlackClient(TOKEN, transport=httpx.MockTransport(handler), sleep=sleeps.append)


def _echo(status: int, **headers):
    """Answer ``status`` with a body that echoes the token, the channel and the URL back."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = {"ok": False, "error": f"bad {TOKEN} {CHANNEL} {request.url}"}
        return httpx.Response(status, json=body, headers=headers)

    return handler


def test_a_call_posts_the_params_as_a_form_with_the_bearer_token():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True, "messages": []})

    assert _client(handler).call("conversations.history", channel=CHANNEL, limit=200) == {
        "ok": True,
        "messages": [],
    }
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == f"{API_URL}conversations.history"
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert parse_qs(request.content.decode()) == {"channel": [CHANNEL], "limit": ["200"]}


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, SlackAuthError),
        (403, SlackAPIError),
        (404, SlackAPIError),
        (429, SlackRateLimitedError),
        (500, SlackAPIError),
        (503, SlackAPIError),
        (302, SlackAPIError),
    ],
)
def test_no_secret_id_or_url_reaches_an_error_or_a_log(caplog, status, error):
    sleeps: list = []
    client = _client(_echo(status, **{"Retry-After": "2", "Location": "https://evil.test"}), sleeps)

    with caplog.at_level(logging.DEBUG), pytest.raises(error) as raised:
        client.call("conversations.history", channel=CHANNEL)

    text = f"{raised.value} {raised.value!r} {raised.value.args!r}"
    for secret in (TOKEN, CHANNEL, "https://", "slack.com"):
        assert secret not in text
    assert raised.value.__cause__ is None
    assert TOKEN not in caplog.text and CHANNEL not in caplog.text
    # httpx itself logs the request line; the URL is the constant endpoint, never an id.
    # Our own lines name the method and Slack's code, never a URL.
    ours = [r.getMessage() for r in caplog.records if not r.name.startswith(("httpx", "httpcore"))]
    assert not any("://" in message for message in ours)


def test_an_error_answer_names_only_the_method_and_slacks_code():
    def handler(request):
        return httpx.Response(200, json={"ok": False, "error": "channel_not_found"})

    with pytest.raises(SlackAPIError) as raised:
        _client(handler).call("conversations.history", channel=CHANNEL)

    assert str(raised.value) == "Slack conversations.history failed: channel_not_found"
    assert raised.value.code == "channel_not_found"


def test_an_error_code_that_is_not_a_slack_code_is_not_echoed():
    with pytest.raises(SlackAPIError) as raised:
        _client(_echo(200)).call("conversations.history", channel=CHANNEL)
    assert raised.value.code == "unknown_error"


@pytest.mark.parametrize("code", ["invalid_auth", "token_revoked", "token_expired", "not_authed"])
def test_a_rejected_token_is_an_auth_error(code):
    def handler(request):
        return httpx.Response(200, json={"ok": False, "error": code})

    with pytest.raises(SlackAuthError):
        _client(handler).call("auth.test")


def test_a_missing_scope_names_the_scope_and_the_reconnect():
    def handler(request):
        return httpx.Response(
            200, json={"ok": False, "error": "missing_scope", "needed": "users:read"}
        )

    with pytest.raises(SlackMissingScopeError) as raised:
        _client(handler).call("users.info", user="U1")
    assert (
        str(raised.value)
        == "Slack users.info needs users:read. Reconnect Slack to grant users:read."
    )
    assert raised.value.code == "missing_scope"


def test_a_200_without_a_json_object_is_an_error_not_an_empty_answer():
    def handler(request):
        return httpx.Response(200, text="<html>proxy</html>")

    with pytest.raises(SlackAPIError, match="invalid_response"):
        _client(handler).call("conversations.history", channel=CHANNEL)


def test_a_rate_limit_waits_retry_after_and_then_succeeds():
    answers = iter(
        [
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json={"ok": False, "error": "ratelimited"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    sleeps: list = []
    assert _client(lambda request: next(answers), sleeps).call("auth.test") == {"ok": True}
    assert sleeps == [7, 1]


def test_a_rate_limit_that_outlasts_the_retries_raises():
    sleeps: list = []
    with pytest.raises(SlackRateLimitedError):
        _client(lambda r: httpx.Response(429, headers={"Retry-After": "3"}), sleeps).call(
            "auth.test"
        )
    assert sleeps == [3, 3, 3], "four attempts, three waits"


@pytest.mark.parametrize("retry_after", ["61", "301"])
def test_a_retry_after_over_60_seconds_stops_at_once_instead_of_sleeping(retry_after):
    sleeps: list = []
    with pytest.raises(SlackRateLimitedError):
        _client(lambda r: httpx.Response(429, headers={"Retry-After": retry_after}), sleeps).call(
            "auth.test"
        )
    assert sleeps == []


def test_tier_1s_minute_is_waited_out():
    """Tier 1 history allows a call a minute: a cut on every 60 s wait synced only the first
    of several channels, on every sync (G46)."""
    answers = iter(
        [httpx.Response(429, headers={"Retry-After": "60"}), httpx.Response(200, json={"ok": True})]
    )
    sleeps: list = []
    assert _client(lambda r: next(answers), sleeps).call("auth.test")["ok"] is True
    assert sleeps == [60]


def test_a_wait_longer_than_the_run_has_left_stops_instead_of_sleeping():
    """The run's own deadline bounds the wait, so the staging lock is not held past it."""
    sleeps: list = []
    with pytest.raises(SlackRateLimitedError):
        _client(lambda r: httpx.Response(429, headers={"Retry-After": "30"}), sleeps).call(
            "auth.test", max_wait=10
        )
    assert sleeps == []


def _history_handler(retry_after: str):
    """auth.test answers; the first history call is rate limited, then history answers."""
    limited = {"done": False}

    def handler(request):
        method = request.url.path.rsplit("/", 1)[1]
        if method == "auth.test":
            body = {"ok": True, "team_id": TEAM, "url": "https://acme.slack.com/"}
            return httpx.Response(200, json=body)
        if method == "conversations.history" and not limited["done"]:
            limited["done"] = True
            return httpx.Response(429, headers={"Retry-After": retry_after})
        if method == "conversations.history":
            body = {
                "ok": True,
                "messages": [{"ts": "1793872860.000100", "user": "U1", "text": "hi"}],
            }
            return httpx.Response(200, json=body)
        if method == "users.info":
            return httpx.Response(200, json={"ok": True, "user": {"name": "ana"}})
        return httpx.Response(200, json={"ok": True, "channel": {"name": "general"}})

    return handler


def _stats() -> dict[str, int]:
    return dict.fromkeys(("scanned", "emitted", "skipped", "deleted", "failed", "requests"), 0)


def test_a_run_rate_limited_for_over_a_minute_is_cut_without_sleeping():
    sleeps: list = []
    stats = _stats()
    rows = list(_iter_rows(_client(_history_handler("61"), sleeps), TEAM, CHANNEL, {}, stats))

    assert rows == [] and sleeps == []
    assert stats["failed_rate_limit"] == 1


def test_a_run_near_its_deadline_is_cut_instead_of_waiting_past_it():
    """10 seconds left of the run and Slack asks for 30: the run stops, the lock is free."""
    sleeps: list = []
    stats = _stats()
    rows = list(
        _iter_rows(
            _client(_history_handler("30"), sleeps),
            TEAM,
            CHANNEL,
            {},
            stats,
            max_seconds=10,
            clock=lambda: 0.0,
        )
    )

    assert rows == [] and sleeps == []
    assert stats["failed_rate_limit"] == 1


def test_a_run_rate_limited_for_5_seconds_waits_and_continues():
    sleeps: list = []
    stats = _stats()
    rows = list(_iter_rows(_client(_history_handler("5"), sleeps), TEAM, CHANNEL, {}, stats))

    assert sleeps == [5]
    # A message without replies is part of its day's document (G7).
    assert [row["id"] for row in rows] == [f"{CHANNEL}:day:2026-11-05"]
    assert stats["failed"] == 0


def test_server_errors_and_network_failures_are_retried_with_a_short_backoff():
    answers = iter(
        [httpx.ConnectError("down"), httpx.Response(502), httpx.Response(200, json={"ok": True})]
    )

    def handler(request):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    sleeps: list = []
    assert _client(handler, sleeps).call("auth.test") == {"ok": True}
    assert sleeps == [1, 2]


@pytest.mark.parametrize("status", [500, 502, 503])
def test_a_persistent_server_error_is_unavailable_after_the_retries(status):
    sleeps: list = []
    with pytest.raises(SlackUnavailableError):
        _client(lambda r: httpx.Response(status), sleeps).call("auth.test")
    assert sleeps == [1, 2, 4]


def test_a_persistent_network_failure_raises_without_the_url():
    def handler(request):
        raise httpx.ConnectError(f"cannot reach {request.url} with {TOKEN}")

    with pytest.raises(SlackUnavailableError) as raised:
        _client(handler).call("conversations.history", channel=CHANNEL)
    assert "slack.com" not in str(raised.value) and TOKEN not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__suppress_context__


@pytest.mark.parametrize("bad", ["", "   ", "xoxb bad", "xoxb-é", "xoxb-\tx"])
def test_a_malformed_token_is_refused_without_echoing_it(bad):
    with pytest.raises(ValueError) as raised:
        SlackClient(bad)
    assert not bad.strip() or bad.strip() not in str(raised.value)


def test_the_token_is_never_shown():
    assert TOKEN not in repr(SlackClient(TOKEN))
    assert TOKEN not in repr(build_slack_service(TOKEN))
    source = slack_source(access_token=TOKEN, team_id=TEAM, channel_id=CHANNEL, resource_name="t")
    assert TOKEN not in repr(source)


def test_a_429_through_the_real_client_ends_the_run_as_failed_rate_limit(caplog):
    def handler(request):
        method = request.url.path.rsplit("/", 1)[1]
        if method == "auth.test":
            return httpx.Response(
                200,
                json={"ok": True, "team_id": TEAM, "url": "https://acme.slack.com/"},
            )
        return httpx.Response(429, headers={"Retry-After": "1"})

    stats = dict.fromkeys(("scanned", "emitted", "skipped", "deleted", "failed", "requests"), 0)
    with caplog.at_level(logging.DEBUG):
        rows = list(_iter_rows(_client(handler), TEAM, CHANNEL, {}, stats))

    assert rows == []
    assert stats["failed"] == 1 and stats["failed_rate_limit"] == 1
    assert TOKEN not in caplog.text and CHANNEL not in caplog.text


class _Revoked(Exception):
    """What a host's check raises once the connection is gone."""


def _revoked_on_second_check():
    checks = iter([None, _Revoked()])

    def check_active():
        failure = next(checks)
        if failure is not None:
            raise failure

    return check_active


@pytest.mark.parametrize(
    "answer",
    [
        {"status_code": 503},
        {"status_code": 429, "headers": {"Retry-After": "5"}},
    ],
    ids=["outage", "rate_limit"],
)
def test_a_revoked_connection_stops_a_retry_before_it_waits(answer):
    """The host's check runs before every attempt and every wait, not once per request."""
    requests, sleeps = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(json={"ok": False, "error": "ratelimited"}, **answer)

    with pytest.raises(_Revoked):
        list(
            _iter_rows(
                _client(handler, sleeps),
                TEAM,
                CHANNEL,
                {},
                _stats(),
                check_active=_revoked_on_second_check(),
            )
        )

    assert len(requests) == 1
    assert sleeps == []
