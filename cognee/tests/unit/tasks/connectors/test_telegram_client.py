# Adapted from topoteretes/cognee-community (Apache-2.0), PR #355, commit 76ff1a4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""The Telegram Bot API client: errors never carry the token, retries stay inside a budget.

The token is part of every request URL, and urllib's ``HTTPError`` keeps that URL. These tests
drive the real client, once through a patched opener and once across a real local socket, and
look for the token in the message, in ``__cause__`` and ``__context__`` and in the logs.
"""

import http.client
import http.server
import io
import json
import math
import threading
import time
import urllib.error

import pytest
from fake_telegram import TOKEN

from cognee.tasks.ingestion.connectors import telegram as module
from cognee.tasks.ingestion.connectors.telegram import (
    TelegramAPIError,
    TelegramAuthError,
    TelegramBotClient,
    TelegramConflictError,
    TelegramRateLimitError,
    TelegramUnavailableError,
    TelegramWebhookConflictError,
    _WaitBudget,
    build_telegram_service,
)

SECRET = TOKEN.split(":", 1)[1]


def _http_error(status, payload, url=f"https://api.telegram.org/bot{TOKEN}/getUpdates"):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return urllib.error.HTTPError(url, status, f"err {url}", {}, io.BytesIO(body))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(outcome if isinstance(outcome, bytes) else json.dumps(outcome).encode())


def _client(*outcomes):
    sleeps = []
    client = TelegramBotClient(TOKEN, sleep=sleeps.append)
    client._open = _Opener(*outcomes)
    return client, sleeps


def _chain(error):
    seen = []
    while error is not None and error not in seen:
        seen.append(error)
        error = error.__cause__ or error.__context__
    return seen


def _assert_no_token(error):
    for link in _chain(error):
        text = f"{link!s} {link!r} {getattr(link, 'description', '')} {getattr(link, 'url', '')}"
        assert SECRET not in text, f"the token leaked through {type(link).__name__}"
    assert error.__cause__ is None and error.__context__ is None


OK = {"ok": True, "result": []}


@pytest.mark.parametrize(
    ("outcomes", "kind"),
    [
        ([_http_error(401, {"description": "Unauthorized"})], TelegramAuthError),
        ([_http_error(404, {"description": "Not Found"})], TelegramAuthError),
        (
            [_http_error(409, {"description": "Conflict: can't use getUpdates while webhook"})],
            TelegramWebhookConflictError,
        ),
        (
            [_http_error(409, {"description": "Conflict: terminated by other getUpdates"})],
            TelegramConflictError,
        ),
        ([_http_error(400, {"description": f"Bad Request: bot{TOKEN}"})], TelegramAPIError),
        ([_http_error(302, b"")], TelegramAPIError),
        ([_http_error(429, {"parameters": {"retry_after": 3600}})], TelegramRateLimitError),
        ([_http_error(502, b"<html>bad gateway</html>")] * 5, TelegramUnavailableError),
        (
            [urllib.error.URLError(f"https://api.telegram.org/bot{TOKEN}")] * 5,
            TelegramUnavailableError,
        ),
        ([ConnectionResetError(104, "reset")] * 5, TelegramUnavailableError),
        ([http.client.RemoteDisconnected("closed")] * 5, TelegramUnavailableError),
        ([http.client.IncompleteRead(b"{")] * 5, TelegramUnavailableError),
        ([TimeoutError()] * 5, TelegramUnavailableError),
        ([b"<html>proxy page</html>"], TelegramAPIError),
        ([{"ok": False, "description": "nope"}], TelegramAPIError),
    ],
)
def test_no_error_carries_the_token_in_its_message_cause_or_context(outcomes, kind, monkeypatch):
    lines = []
    monkeypatch.setattr(
        module.logger, "warning", lambda fmt, *a: lines.append(fmt % a if a else fmt)
    )
    client, _ = _client(*outcomes)
    with pytest.raises(kind) as info:
        client.call("getUpdates")
    _assert_no_token(info.value)
    assert not any(SECRET in line for line in lines)


def test_a_409_says_whether_a_webhook_or_another_poller_is_the_cause():
    client, _ = _client(_http_error(409, {"description": "Conflict: webhook is active"}))
    with pytest.raises(TelegramWebhookConflictError, match="webhook"):
        client.call("getUpdates")
    client, _ = _client(_http_error(409, {"description": "Conflict: terminated by other"}))
    with pytest.raises(TelegramConflictError, match="another process") as info:
        client.call("getUpdates")
    assert not isinstance(info.value, TelegramWebhookConflictError)


def test_short_flood_waits_are_slept_long_ones_stop_without_sleeping():
    client, sleeps = _client(_http_error(429, {"parameters": {"retry_after": 5}}), OK)
    assert client.call("getUpdates") == []
    assert sleeps == [5.0]

    client, sleeps = _client(_http_error(429, {"parameters": {"retry_after": 61}}))
    with pytest.raises(TelegramRateLimitError) as info:
        client.call("getUpdates")
    assert info.value.retry_after == 61 and sleeps == []


def test_the_wait_budget_is_shared_by_every_request_of_a_run():
    budget = _WaitBudget()
    flood = {"parameters": {"retry_after": 40}}
    client, sleeps = _client(_http_error(429, flood), OK, _http_error(429, flood))
    assert client.call("getMe", budget=budget) == []
    with pytest.raises(TelegramRateLimitError):
        client.call("getUpdates", budget=budget)
    assert sleeps == [40.0], "the second 40 s wait does not fit the 60 s a run may sleep"


def test_a_5xx_is_retried_with_backoff_and_then_raised():
    client, sleeps = _client(*[_http_error(500, {})] * 5)
    with pytest.raises(TelegramUnavailableError):
        client.call("getUpdates")
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


@pytest.mark.parametrize(
    "blip",
    [
        ConnectionResetError(104, "reset"),
        http.client.RemoteDisconnected("closed"),
        http.client.IncompleteRead(b"{"),
        TimeoutError(),
    ],
)
def test_a_dropped_connection_is_retried(blip):
    client, sleeps = _client(blip, OK)
    assert client.call("getUpdates") == []
    assert sleeps == [1.0]


def test_check_active_runs_before_every_attempt_and_wait_and_stops_a_backoff():
    class Revoked(Exception):
        pass

    events = []
    client = TelegramBotClient(TOKEN, sleep=lambda s: events.append(("sleep", s)))
    opener = _Opener(*[_http_error(500, {})] * 5)
    client._open = lambda request: (events.append(("request",)), opener(request))[1]

    def check_active():
        events.append(("check",))
        if sum(event == ("sleep", 1.0) for event in events):
            raise Revoked

    with pytest.raises(Revoked) as info:
        client.call("getUpdates", check_active=check_active)
    assert events == [("check",), ("request",), ("check",), ("sleep", 1.0), ("check",)]
    assert info.value.__context__ is None, "what check_active raises propagates as it is"


def test_a_redirect_is_never_followed(monkeypatch):
    server, seen = _serve({"/redirect": (302, b"", {"Location": "http://example.invalid/"})})
    monkeypatch.setattr(module, "_API_BASE", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(module, "TelegramBotClient", TelegramBotClient)
    client = TelegramBotClient(TOKEN, sleep=lambda _: None)
    try:
        with pytest.raises(TelegramAPIError) as info:
            client.call("redirect")
    finally:
        server.shutdown()
    assert info.value.status == 302
    assert seen == [f"/bot{TOKEN}/redirect"], "one request, to the bot's own path"
    _assert_no_token(info.value)


def test_a_real_socket_round_trip_answers_and_rejects(monkeypatch):
    server, seen = _serve(
        {
            "/getMe": (200, json.dumps({"ok": True, "result": {"id": 111}}).encode(), {}),
            "/getUpdates": (
                401,
                json.dumps({"ok": False, "description": "Unauthorized"}).encode(),
                {},
            ),
        }
    )
    monkeypatch.setattr(module, "_API_BASE", f"http://127.0.0.1:{server.server_port}")
    client = build_telegram_service(TOKEN)
    try:
        assert client.call("getMe") == {"id": 111}
        with pytest.raises(TelegramAuthError) as info:
            client.call("getUpdates", {"offset": 5, "limit": 10, "timeout": None})
    finally:
        server.shutdown()
    _assert_no_token(info.value)
    assert seen == [f"/bot{TOKEN}/getMe", f"/bot{TOKEN}/getUpdates"]


@pytest.mark.parametrize(
    ("status", "body", "key", "raises"),
    [
        (503, b"<html>maintenance</html>", "failed_unavailable", False),
        (400, json.dumps({"ok": False, "description": f"Bad bot{TOKEN}"}).encode(), None, True),
    ],
)
def test_a_real_socket_run_stops_on_a_5xx_and_raises_on_another_error(
    status, body, key, raises, monkeypatch
):
    """Through the source: a 5xx after the retries stops the run with ``failed_unavailable``,
    another error answer raises with ``failed_provider``, and the token is in neither."""
    from cognee.tasks.ingestion.connectors import telegram_source

    server, seen = _serve(
        {
            "/getMe": (200, json.dumps({"ok": True, "result": {"id": 111}}).encode(), {}),
            "/getUpdates": (status, body, {}),
        }
    )
    lines = []
    monkeypatch.setattr(
        module.logger, "warning", lambda fmt, *a: lines.append(fmt % a if a else fmt)
    )
    monkeypatch.setattr(module, "_API_BASE", f"http://127.0.0.1:{server.server_port}")
    client = TelegramBotClient(TOKEN, sleep=lambda _: None)
    resource = telegram_source(service=client, resource_name="telegram_111")
    try:
        if raises:
            with pytest.raises(Exception) as info:
                list(resource)
            for link in _chain(info.value):
                assert SECRET not in f"{link!s} {link!r} {getattr(link, 'description', '')}"
        else:
            assert list(resource) == []
    finally:
        server.shutdown()
    stats = resource.cognee_sync_stats
    assert stats["failed"] == 1
    assert stats[key or "failed_provider"] == 1
    assert stats["failed_unavailable" if raises else "failed_provider"] == 0
    assert seen.count(f"/bot{TOKEN}/getUpdates") == (1 if raises else 5)
    assert not any(SECRET in line for line in lines)


def _serve(routes):
    """A real HTTP server on port 0 answering ``routes[<method suffix>] = (status, body, headers)``."""
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            status, body, headers = routes["/" + self.path.rsplit("/", 1)[1]]
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


@pytest.mark.parametrize(
    "token", ["", "  ", "abc", "123", "123:short", "bot123:" + "a" * 30, "12:" + "a" * 30 + "\n1"]
)
def test_a_token_of_the_wrong_shape_is_refused_without_echoing_it(token):
    with pytest.raises(ValueError, match="invalid format") as info:
        TelegramBotClient(token)
    if token.strip():
        assert token.strip() not in str(info.value)


def test_the_client_repr_hides_the_token():
    assert SECRET not in repr(TelegramBotClient(TOKEN))
    assert TelegramBotClient(TOKEN).bot_id == "111"


@pytest.mark.parametrize("raw", [b"-1", b"-Infinity", b"true", b'"5"'])
def test_a_429_with_an_unusable_retry_after_falls_back_to_backoff(raw):
    """Two 429s, so the waits are 1 then 2: a value read as a wait (``true`` as 1.0) would give
    1 then 1."""
    body = b'{"ok": false, "parameters": {"retry_after": ' + raw + b"}}"
    client, sleeps = _client(
        _http_error(429, body), _http_error(429, body), {"ok": True, "result": []}
    )
    assert client.call("getUpdates") == []
    assert sleeps == [1.0, 2.0]


def test_a_429_with_a_nan_retry_after_falls_back_to_backoff():
    """Python's json reads NaN, and a NaN wait is no number of seconds at all."""
    body = b'{"ok": false, "parameters": {"retry_after": NaN}}'
    client, sleeps = _client(_http_error(429, body), {"ok": True, "result": []})
    assert client.call("getUpdates") == []
    assert sleeps == [1.0]


@pytest.mark.parametrize("raw", [b"Infinity", b"1e400", b"1" + b"0" * 309, b"1" + b"0" * 400])
def test_a_429_whose_wait_is_beyond_any_number_stops_without_sleeping(raw):
    """A wait longer than the largest float is longer than any budget, like 1e308: one request,
    no sleep, and the rate limit raised for the run to stop on."""
    body = b'{"ok": false, "parameters": {"retry_after": ' + raw + b"}}"
    client, sleeps = _client(_http_error(429, body), {"ok": True, "result": []})
    with pytest.raises(TelegramRateLimitError) as info:
        client.call("getUpdates")
    assert info.value.retry_after == math.inf
    assert sleeps == [] and len(client._open.requests) == 1


def test_the_largest_finite_retry_after_is_kept_and_stops_without_sleeping():
    """A boundary guard, not a reproduction: the largest float is a wait, kept as it is."""
    body = b'{"ok": false, "parameters": {"retry_after": 1.7976931348623157e308}}'
    client, sleeps = _client(_http_error(429, body))
    with pytest.raises(TelegramRateLimitError) as info:
        client.call("getUpdates")
    assert info.value.retry_after == 1.7976931348623157e308 and sleeps == []


def test_a_retry_after_of_0_is_a_wait_of_0_seconds():
    """A boundary guard: 0 is a valid wait, not a reason to back off."""
    body = b'{"ok": false, "parameters": {"retry_after": 0}}'
    client, sleeps = _client(_http_error(429, body), {"ok": True, "result": []})
    assert client.call("getUpdates") == []
    assert sleeps == [0.0]


@pytest.mark.parametrize(
    "update",
    [
        (
            b'{"update_id": NaN, "message": {"message_id": 1, "date": 1700000000, "text": "a",'
            b' "chat": {"id": -100, "type": "group", "title": "T"}}}'
        ),
        (
            b'{"update_id": 1000, "message": {"message_id": 1, "date": NaN, "text": "a",'
            b' "chat": {"id": -100, "type": "group", "title": "T"}}}'
        ),
    ],
    ids=["update_id", "date"],
)
def test_a_nan_in_a_getupdates_answer_on_the_wire_fails_as_a_provider_defect(update, monkeypatch):
    """A getUpdates answer whose JSON carries NaN for update_id or date, read through the
    real client and a real socket, fails the run as failed_provider and emits nothing."""
    from cognee.tasks.ingestion.connectors import telegram_source

    server, _ = _serve(
        {
            "/getMe": (200, json.dumps({"ok": True, "result": {"id": 111}}).encode(), {}),
            "/getUpdates": (200, b'{"ok": true, "result": [' + update + b"]}", {}),
        }
    )
    monkeypatch.setattr(module, "_API_BASE", f"http://127.0.0.1:{server.server_port}")
    resource = telegram_source(
        service=TelegramBotClient(TOKEN, sleep=lambda _: None), resource_name="telegram_111"
    )
    try:
        # dlt wraps what the resource raised; the host reads the reason through the chain.
        with pytest.raises(Exception) as info:
            list(resource)
    finally:
        server.shutdown()
    assert module.classify_sync_error(info.value) == "failed_provider"
    stats = resource.cognee_sync_stats
    assert [key for key, value in stats.items() if value] == ["failed", "failed_provider"]


def test_a_run_whose_429s_name_a_nan_wait_stops_as_a_rate_limit(monkeypatch):
    """Through the source and a real socket: every getUpdates answers 429 with a NaN wait, so
    the run backs off 1, 2, 4 and 8 s and stops with failed_rate_limit, not failed_unexpected.
    The recorded sleep still passes each wait through time.sleep's own argument check (scaled
    to 0 s), so a NaN reaching it raises as it would in a real run."""
    from cognee.tasks.ingestion.connectors import telegram_source

    flood = b'{"ok": false, "error_code": 429, "parameters": {"retry_after": NaN}}'
    server, seen = _serve(
        {
            "/getMe": (200, json.dumps({"ok": True, "result": {"id": 111}}).encode(), {}),
            "/getUpdates": (429, flood, {}),
        }
    )
    monkeypatch.setattr(module, "_API_BASE", f"http://127.0.0.1:{server.server_port}")
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        time.sleep(0 * seconds)

    client = TelegramBotClient(TOKEN, sleep=sleep)
    resource = telegram_source(service=client, resource_name="telegram_111")
    try:
        assert list(resource) == []
    finally:
        server.shutdown()
    stats = resource.cognee_sync_stats
    assert [key for key, value in stats.items() if value] == ["failed", "failed_rate_limit"]
    assert sleeps == [1.0, 2.0, 4.0, 8.0]
    assert seen.count(f"/bot{TOKEN}/getUpdates") == 5


def test_a_5xx_whose_body_cannot_be_read_is_unavailable_and_retried():
    class CutOff(urllib.error.HTTPError):
        def read(self, *args):
            raise http.client.IncompleteRead(b"")

    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    error = CutOff(url, 502, "bad gateway", {}, io.BytesIO(b""))
    client, sleeps = _client(error, {"ok": True, "result": [1]})
    assert client.call("getUpdates") == [1]
    assert sleeps == [1.0]
    client, _ = _client(*[error] * 5)
    with pytest.raises(TelegramUnavailableError) as info:
        client.call("getUpdates")
    assert info.value.status == 502
    _assert_no_token(info.value)


def test_chat_not_found_on_the_wire_resolves_a_username_to_nothing():
    """The real client reads Telegram's 400 answer for an unknown @username; that answer, and
    no other, means the name names no chat."""
    not_found = {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
    client, sleeps = _client(_http_error(400, not_found))
    selection = module._selection(["@news_typo"], False)
    assert module.resolve_usernames(client, {}, selection, module._StoredMessages(list)) == {
        "news_typo": None
    }
    assert sleeps == []
    other = {"ok": False, "error_code": 400, "description": "Bad Request: something else"}
    client, _ = _client(_http_error(400, other))
    with pytest.raises(TelegramAPIError) as info:
        module.resolve_usernames(client, {}, selection, module._StoredMessages(list))
    _assert_no_token(info.value)


def test_the_account_id_is_the_bot_id_the_source_checks_its_state_against():
    """A host keys a connection by this id, so a second bot never inherits the first bot's
    table and state (which the source would refuse with failed_bot_mismatch on every run)."""
    client, _ = _client({"ok": True, "result": {"id": 123456789, "is_bot": True}})
    assert module.telegram_account_id(client) == "123456789"
    assert client._open.requests[0].full_url.endswith("/getMe")


@pytest.mark.parametrize(
    "answer",
    [{"ok": True, "result": {"is_bot": True}}, {"ok": True, "result": {"id": True}}],
)
def test_an_account_id_answer_without_the_bot_id_raises(answer):
    client, _ = _client(answer)
    with pytest.raises(TelegramAPIError):
        module.telegram_account_id(client)


def test_a_rejected_token_raises_on_the_account_id_call_without_the_token():
    client, _ = _client(_http_error(401, {"ok": False, "description": "Unauthorized"}))
    with pytest.raises(TelegramAuthError) as info:
        module.telegram_account_id(client)
    _assert_no_token(info.value)


def test_the_account_id_call_checks_the_host_before_its_request():
    """F37: a host can stop the account id call like every other request."""
    calls = []
    client, _ = _client({"ok": True, "result": {"id": 5, "is_bot": True}})
    assert module.telegram_account_id(client, check_active=lambda: calls.append(1)) == "5"
    assert calls == [1]
