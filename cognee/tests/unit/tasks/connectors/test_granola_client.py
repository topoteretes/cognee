# Adapted from topoteretes/cloud-backend, CLO-504, cognee-saas-pod/tests/unit/test_granola_source.py at aa05be08.
# Modified to ship in the Cognee SDK: the client tests, run through requests with a faked transport.
"""The Granola client through the real ``requests`` stack, with ``Session.send`` faked.

Covers the wire: the fixed origin and path set, no redirects, the Bearer key (netrc cannot
replace it), the body cap, one immediate retry on a 5xx or no answer, pacing, ``Retry-After``,
``check_active`` before every attempt, and that no error or log line carries the key, a note
id or a URL, whatever Granola answers.
"""

import json
import logging
from datetime import datetime, timezone

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from cognee.tasks.ingestion.connectors import granola
from cognee.tasks.ingestion.connectors.granola import (
    GranolaClient,
    GranolaError,
    GranolaNoteError,
    _iter_rows,
    build_granola_service,
    retry_after_seconds,
)

KEY = "grn_SECRETKEY0123456789"
NOTE = "not_123456789ABCDE"
FOLDER = "fol_123456789ABCDE"
NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)
URL_PART = "public-api.granola.ai"


def answer(status: int, body=b"", headers=None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    response._content_consumed = True
    response.headers = CaseInsensitiveDict(headers or {})
    return response


class Transport:
    """Stands in for ``Session.send``: records what requests really sends, answers in turn."""

    def __init__(self, monkeypatch, *answers):
        self.answers = list(answers)
        self.sent: list[tuple[requests.PreparedRequest, dict]] = []
        transport = self

        def send(session, request, **kwargs):
            transport.sent.append((request, kwargs))
            result = transport.answers.pop(0)
            if isinstance(result, Exception):
                raise result
            result.url = request.url
            return result

        monkeypatch.setattr(requests.sessions.Session, "send", send)


def client(**kwargs) -> GranolaClient:
    kwargs.setdefault("sleep", lambda _seconds: None)
    return GranolaClient(KEY, **kwargs)


def test_the_key_goes_out_as_a_bearer_header_to_the_fixed_origin(monkeypatch):
    wire = Transport(monkeypatch, answer(200, {"notes": [], "hasMore": False}))

    client().get("/v1/notes", {"page_size": 1})

    request, kwargs = wire.sent[0]
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.url == "https://public-api.granola.ai/v1/notes?page_size=1"
    assert kwargs["allow_redirects"] is False and kwargs["stream"] is True


def test_a_key_pasted_with_surrounding_whitespace_is_sent_stripped(monkeypatch):
    wire = Transport(monkeypatch, answer(200, {"notes": [], "hasMore": False}))

    GranolaClient(f" {KEY}\n", sleep=lambda _seconds: None).get("/v1/notes")

    assert wire.sent[0][0].headers["Authorization"] == f"Bearer {KEY}"


def test_a_netrc_entry_for_the_host_never_replaces_the_bearer_key(monkeypatch, tmp_path):
    """requests reads ~/.netrc when no auth object is given and overwrites an Authorization
    header with it; the client passes an auth object, so the key is what goes out."""
    netrc = tmp_path / "netrc"
    netrc.write_text("machine public-api.granola.ai login someone password other\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    wire = Transport(monkeypatch, answer(200, {}))

    client().get("/v1/notes")

    assert wire.sent[0][0].headers["Authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize(
    "path",
    [
        "https://attacker.example/v1/notes",
        "/v1/notes/../users",
        "/v1/notes/not_123456789ABCDE/../../x",
        "/v1/notes?cursor=x",
        "/v1/notes/not_short",
    ],
)
def test_only_the_three_note_paths_can_be_requested(monkeypatch, path):
    wire = Transport(monkeypatch)
    with pytest.raises(ValueError, match="Invalid Granola API path"):
        client().get(path)
    assert wire.sent == []


def test_a_redirect_is_an_error_and_is_never_followed(monkeypatch):
    wire = Transport(
        monkeypatch, answer(302, {"secret": KEY}, {"Location": "https://attacker.example/"})
    )
    with pytest.raises(GranolaError) as raised:
        client().get("/v1/notes")
    assert raised.value.status == 302 and KEY not in str(raised.value)
    assert len(wire.sent) == 1


def test_a_body_over_the_cap_is_refused(monkeypatch):
    monkeypatch.setattr(granola, "MAX_BODY", 64)
    Transport(monkeypatch, answer(200, {"notes": ["x" * 100], "hasMore": False}))
    with pytest.raises(GranolaNoteError, match="size limit"):
        client().get("/v1/notes")


@pytest.mark.parametrize("body", [b"<html>proxy</html>", b"[]", b"\xff\xfe"])
def test_a_200_that_is_not_a_json_object_is_an_invalid_response(monkeypatch, body):
    Transport(monkeypatch, answer(200, body))
    with pytest.raises(GranolaNoteError, match="Invalid Granola response"):
        client().get("/v1/notes")


def test_an_unreadable_body_is_no_errors_context(monkeypatch):
    """A JSON error keeps the body in ``.doc``; raised inside the ``except`` it would ride
    along as ``__context__`` to anyone calling the client (review X3)."""
    Transport(monkeypatch, answer(200, f"not json {KEY} {NOTE}".encode()))
    with pytest.raises(GranolaNoteError) as raised:
        client().get(f"/v1/notes/{NOTE}")
    assert raised.value.__context__ is None and raised.value.__cause__ is None

    broken = answer(200)

    def iter_content(_size):
        raise requests.exceptions.ChunkedEncodingError(f"https://{URL_PART}/v1/notes/{NOTE}")

    broken.iter_content = iter_content
    Transport(monkeypatch, broken)
    with pytest.raises(GranolaError, match="Could not read") as raised:
        client().get(f"/v1/notes/{NOTE}")
    assert raised.value.__context__ is None


@pytest.fixture
def local_granola(monkeypatch):
    """A real HTTP server on 127.0.0.1 in place of Granola: ``routes`` maps a query string to
    ``(status, headers, body chunks)``; any other request answers ``{"id": NOTE}``. The body
    goes over a real socket, so the client's ``stream=True`` read is what is tested."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    routes: dict[str, tuple[int, dict, list[bytes]]] = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            query = self.path.partition("?")[2]
            status, headers, chunks = routes.get(
                query, (200, {}, [json.dumps({"id": NOTE}).encode()])
            )
            self.send_response(status)
            chunked = len(chunks) > 1
            for name, value in headers.items():
                self.send_header(name, value)
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(chunks[0]) if chunks else 0))
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk) if chunked else chunk)
            if chunked:
                self.wfile.write(b"0\r\n\r\n")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(granola, "API_BASE", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    yield routes
    server.shutdown()
    server.server_close()


def test_urllib3_debug_lines_about_granola_never_name_a_note(local_granola, caplog):
    """urllib3 logs every request line at DEBUG, path and all; a real socket proves the
    client's filter drops them."""
    with caplog.at_level(logging.DEBUG, logger="urllib3"):
        assert client().get(f"/v1/notes/{NOTE}") == {"id": NOTE}
    assert NOTE not in caplog.text and KEY not in caplog.text


@pytest.mark.parametrize(
    "query,answer_,status,retry_after,error",
    [
        ("big=1", (200, {}, [b"x" * 100] * 20), None, 0, granola.GranolaTooLargeError),
        ("moved=1", (302, {"Location": "http://127.0.0.1:1/"}, [b""]), 302, 0, GranolaError),
        ("busy=1", (429, {"Retry-After": "7"}, [b"{}"]), 429, 7, GranolaError),
        ("huge=1", (413, {}, [b"{}"]), 413, 0, GranolaError),
        ("down=1", (500, {}, [b"{}"]), 500, 0, GranolaError),
    ],
)
def test_the_streamed_read_over_a_real_socket(
    local_granola, monkeypatch, query, answer_, status, retry_after, error
):
    """The faked transport hands back bodies already read; these go through ``iter_content``:
    a chunked body over the cap, a redirect not followed, ``Retry-After``, a 413 and a 5xx."""
    monkeypatch.setattr(granola, "MAX_BODY", 500)
    local_granola[query] = answer_
    with pytest.raises(error) as raised:
        client().get("/v1/notes", dict([query.split("=")]))
    assert type(raised.value) is error and raised.value.status == status
    assert raised.value.retry_after == retry_after
    assert KEY not in str(raised.value) and "127.0.0.1" not in str(raised.value)


def test_a_5xx_is_tried_once_more_at_once_and_then_raised(monkeypatch):
    wire = Transport(monkeypatch, answer(503), answer(502))
    with pytest.raises(GranolaError) as raised:
        client().get("/v1/notes")
    assert raised.value.status == 502 and len(wire.sent) == 2

    wire = Transport(monkeypatch, answer(500), answer(200, {"ok": True}))
    assert client().get("/v1/notes") == {"ok": True}


def test_no_retry_once_the_run_deadline_has_passed(monkeypatch):
    wire = Transport(monkeypatch, answer(503), answer(200, {}))
    with pytest.raises(GranolaError):
        client().get("/v1/notes", time_left=lambda: 0)
    assert len(wire.sent) == 1


def test_a_network_error_names_its_class_only(monkeypatch):
    error = requests.ConnectionError(f"https://public-api.granola.ai/v1/notes/{NOTE}?x={KEY}")
    Transport(monkeypatch, error, error)
    with pytest.raises(GranolaError) as raised:
        client().get(f"/v1/notes/{NOTE}")
    assert str(raised.value) == "Granola request failed: ConnectionError"
    assert raised.value.__context__ is None and raised.value.__cause__ is None


def test_check_active_runs_before_every_attempt_retries_included(monkeypatch):
    Transport(monkeypatch, answer(503), answer(200, {}))
    asked = []
    client().get("/v1/notes", check_active=lambda: asked.append(1))
    # Before the first attempt, then (dlt_utils.with_retries) before the wait and the retry.
    assert len(asked) == 3


def test_requests_are_paced_without_sleeping_when_already_slow(monkeypatch):
    Transport(monkeypatch, *[answer(200, {}) for _ in range(3)])
    now = [0.0]
    sleeps: list[float] = []
    paced = GranolaClient(KEY, sleep=sleeps.append, clock=lambda: now[0])
    paced.get("/v1/notes")
    assert sleeps == []  # the first request never waits
    now[0] += 0.05
    paced.get("/v1/notes")
    assert sleeps == [pytest.approx(0.20)]
    now[0] += 1.0
    paced.get("/v1/notes")
    assert len(sleeps) == 1  # slower than the limit already: no sleep


@pytest.mark.parametrize(
    "header,expected",
    [
        ("120", 120),
        ("0.5", 1),
        ("-1", 0),
        ("NaN", 0),
        ("inf", 0),
        (None, 0),
        ("invalid", 0),
        ("999999999999", 604800),
        ("Thu, 12 Nov 2026 00:10:00 GMT", 600),
    ],
)
def test_retry_after_accepts_seconds_or_http_date_and_bounds_bad_values(header, expected):
    assert retry_after_seconds(header, now=NOW) == expected


def test_a_429_carries_its_retry_after_and_is_not_retried(monkeypatch):
    wire = Transport(monkeypatch, answer(429, b"", {"Retry-After": "600"}))
    with pytest.raises(GranolaError) as raised:
        client().get("/v1/notes")
    assert raised.value.status == 429 and raised.value.retry_after == 600
    assert len(wire.sent) == 1


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
def test_no_error_or_log_line_carries_the_key_a_note_id_or_a_url(monkeypatch, caplog, status):
    """Through the walker and the real client: a listing that names the note, then the note's
    read answers ``status`` with a body that quotes the key, the id and the URL."""
    body = {"error": f"{KEY} cannot read {NOTE} at https://{URL_PART}/v1/notes/{NOTE}"}
    listing = answer(200, {"notes": [{"id": NOTE}], "hasMore": False, "cursor": None})
    probe = answer(200, {"notes": [], "hasMore": False, "cursor": None})
    Transport(monkeypatch, listing, answer(status, body), answer(status, body), probe, probe)
    state: dict = {"reconciled_at": NOW.isoformat()}
    stats: dict = {}

    with caplog.at_level(logging.DEBUG):
        rows = list(_iter_rows(client(), FOLDER, state, stats, now=NOW))
        with pytest.raises(GranolaError) as raised:
            Transport(monkeypatch, answer(status, body), answer(status, body))
            client().get(f"/v1/notes/{NOTE}")

    assert rows == [] and stats["failed"] >= 1
    for secret in (KEY, NOTE, URL_PART, FOLDER):
        assert secret not in str(raised.value) and secret not in repr(raised.value)
        assert secret not in caplog.text
    # State keeps note ids by design (positions, retries, ``known``), never the key.
    assert KEY not in json.dumps(state) and URL_PART not in json.dumps(state)


def test_the_repr_hides_the_key():
    assert KEY not in repr(build_granola_service(KEY))
    assert repr(build_granola_service(KEY)) == "GranolaClient(api_key=<redacted>)"
    assert KEY not in repr(granola.dlt_utils.BearerAuth(KEY))


@pytest.mark.parametrize("key", ["", "grn with space", "grn_é", None, 42])
def test_a_malformed_key_is_refused_without_being_echoed(key):
    with pytest.raises(ValueError, match="invalid format") as raised:
        GranolaClient(key)
    assert "with space" not in str(raised.value)


def test_a_5xx_answer_is_closed_before_the_retry(monkeypatch):
    first = answer(503)
    closed = []
    first.close = lambda: closed.append(True)
    Transport(monkeypatch, first, answer(200, {}))
    client().get("/v1/notes")
    assert closed == [True]
