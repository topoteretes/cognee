# Adapted from topoteretes/cognee-community (Apache-2.0), PR #286, commit 0d9c0a7 (itssaharsh).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""The real client over a real socket: a local HTTP server stands in for api.vercel.com.

Everything else in the suite fakes ``requests.request`` or the ``service=`` surface; here the
bytes go through requests, urllib3 and a TCP connection to ``127.0.0.1``, and the assertions are
on what the server received and answered. The base URL and the timeouts are the module's own
constants, patched for the test: no keyword or environment variable on the public client exists
only for tests.
"""

import http.server
import json
import threading
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fake_vercel import FakeVercel

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.connectors import vercel as vercel_module
from cognee.tasks.ingestion.connectors.vercel import (
    DEPLOYMENT_ROUTE,
    DEPLOYMENTS_ROUTE,
    EVENTS_ROUTE,
    PROJECT_ROUTE,
    VercelAPIError,
    VercelClient,
    VercelNotFoundError,
    VercelRateLimitedError,
    VercelResponseError,
    new_stats,
    vercel_source,
)

pytestmark = pytest.mark.timeout(60)

TOKEN = "SECRET-wire-token-0001"
TEAM = "team_wireacme"
PROJECT = "prj_wireweb"
SECRETS = (TOKEN, TEAM, PROJECT, "127.0.0.1", "http://")


class _Server:
    """A local HTTP server that records every request and answers from a script.

    ``script[path]`` is a list of answers used in order (the last one repeats): a tuple
    ``(status, body bytes, headers)``, ``"close"`` (the connection is closed with no answer) or
    ``("sleep", seconds)`` (answers 200 after that long). ``fake`` answers every path the script
    does not name, the way Vercel's API does.
    """

    def __init__(self):
        self.requests: list[dict] = []
        self.script: dict[str, list] = {}
        self.fake: FakeVercel | None = None
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the test output quiet
                pass

            def do_GET(self):
                split = urlsplit(self.path)
                server.requests.append(
                    {
                        "path": split.path,
                        "query": parse_qs(split.query),
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                    }
                )
                answers = server.script.get(split.path)
                if answers:
                    answer = answers.pop(0) if len(answers) > 1 else answers[0]
                elif server.fake is not None:
                    answer = server._fake_answer(split.path, parse_qs(split.query))
                else:
                    answer = (404, b"{}", {})
                if answer == "close":
                    self.close_connection = True
                    return
                if answer[0] == "sleep":
                    time.sleep(answer[1])
                    answer = (200, b"{}", {})
                status, body, headers = answer
                self.send_response(status)
                for name, value in {"Content-Type": "application/json", **headers}.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def _fake_answer(self, path: str, query: dict) -> tuple:
        params: dict = {}
        for key, values in query.items():
            value = values[-1]
            params[key] = int(value) if value.lstrip("-").isdigit() else value
        status, body = self.fake.answer(path, params)
        return status, json.dumps(body).encode(), {}

    def paths(self) -> list[str]:
        return [request["path"] for request in self.requests]


@pytest.fixture
def server(monkeypatch):
    wire = _Server()
    wire.thread.start()
    monkeypatch.setattr(vercel_module, "BASE_URL", wire.base)
    yield wire
    wire.httpd.shutdown()
    wire.httpd.server_close()
    wire.thread.join(timeout=10)


def _client() -> VercelClient:
    return VercelClient(TOKEN, sleep=lambda _seconds: None)


def _json(status: int, body) -> tuple:
    return status, json.dumps(body).encode(), {}


def _chain(error: BaseException) -> list[BaseException]:
    seen = []
    while error is not None and error not in seen:
        seen.append(error)
        error = error.__cause__ or error.__context__
    return seen


def _assert_nothing_named(error: BaseException) -> None:
    for link in _chain(error):
        for secret in SECRETS:
            assert secret not in str(link), (secret, link)
            assert secret not in repr(link), (secret, link)


def test_the_project_read_goes_out_with_the_bearer_token_and_the_team(
    server, monkeypatch, tmp_path
):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login someone password netrc-secret\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    server.script["/v9/projects/" + PROJECT] = [_json(200, {"id": PROJECT})]

    answer = _client().request(PROJECT_ROUTE, ids={"project": PROJECT}, params={"teamId": TEAM})

    assert answer == {"id": PROJECT}
    (received,) = server.requests
    assert received["path"] == f"/v9/projects/{PROJECT}"
    assert received["query"] == {"teamId": [TEAM]}
    assert received["headers"]["authorization"] == f"Bearer {TOKEN}", "netrc did not replace it"
    assert received["headers"]["accept"] == "application/json"
    assert "cookie" not in received["headers"]


def test_a_listing_page_and_the_events_read_send_their_query(server):
    server.script[DEPLOYMENTS_ROUTE] = [
        _json(200, {"deployments": [], "pagination": {"count": 0, "next": None, "prev": None}})
    ]
    server.script["/v3/deployments/dpl_x/events"] = [_json(200, [])]
    client = _client()

    client.request(
        DEPLOYMENTS_ROUTE,
        params={"projectId": PROJECT, "limit": 100, "until": 1794312000001, "teamId": TEAM},
    )
    client.request(
        EVENTS_ROUTE, ids={"deployment": "dpl_x"}, params={"limit": -1, "follow": 0, "teamId": TEAM}
    )

    listing, events = server.requests
    assert listing["path"] == "/v7/deployments"
    assert listing["query"] == {
        "projectId": [PROJECT],
        "limit": ["100"],
        "until": ["1794312000001"],
        "teamId": [TEAM],
    }
    assert events["path"] == "/v3/deployments/dpl_x/events"
    assert events["query"] == {"limit": ["-1"], "follow": ["0"], "teamId": [TEAM]}
    for received in (listing, events):
        assert received["headers"]["authorization"] == f"Bearer {TOKEN}"
        assert "cookie" not in received["headers"]


def test_a_redirect_is_an_error_and_is_never_followed(server):
    server.script["/v13/deployments/dpl_x"] = [(302, b"", {"Location": "/target"})]

    with pytest.raises(VercelAPIError, match="HTTP 302") as raised:
        _client().request(DEPLOYMENT_ROUTE, ids={"deployment": "dpl_x"})

    assert server.paths() == ["/v13/deployments/dpl_x"], "the server never saw /target"
    _assert_nothing_named(raised.value)


def test_a_404_is_not_found(server):
    server.script["/v13/deployments/dpl_x"] = [_json(404, {"error": {"code": "not_found"}})]

    with pytest.raises(VercelNotFoundError, match="HTTP 404") as raised:
        _client().request(DEPLOYMENT_ROUTE, ids={"deployment": "dpl_x"}, params={"teamId": TEAM})

    _assert_nothing_named(raised.value)


def test_a_429_is_not_retried(server):
    server.script["/v13/deployments/dpl_x"] = [(429, b"{}", {"Retry-After": "5"})]

    with pytest.raises(VercelRateLimitedError, match="HTTP 429"):
        _client().request(DEPLOYMENT_ROUTE, ids={"deployment": "dpl_x"})

    assert len(server.requests) == 1


def test_a_503_then_a_200_is_retried_and_succeeds(server):
    server.script["/v13/deployments/dpl_x"] = [_json(503, {}), _json(200, {"id": "dpl_x"})]

    assert _client().request(DEPLOYMENT_ROUTE, ids={"deployment": "dpl_x"}) == {"id": "dpl_x"}
    assert len(server.requests) == 2


def test_a_connection_closed_without_an_answer_ends_in_no_answer_after_the_retries(server):
    server.script["/v13/deployments/dpl_x"] = ["close"]

    with pytest.raises(VercelAPIError, match="no answer") as raised:
        _client().request(DEPLOYMENT_ROUTE, ids={"deployment": "dpl_x"}, params={"teamId": TEAM})

    assert raised.value.status == 0
    assert len(server.requests) == 3
    _assert_nothing_named(raised.value)


def test_a_read_timeout_ends_in_no_answer_after_the_retries(server, monkeypatch):
    monkeypatch.setattr(vercel_module, "_TIMEOUT", (2, 0.2))
    server.script["/v13/deployments/dpl_x"] = [("sleep", 1.0)]

    with pytest.raises(VercelAPIError, match="no answer") as raised:
        _client().request(DEPLOYMENT_ROUTE, ids={"deployment": "dpl_x"}, params={"teamId": TEAM})

    assert raised.value.status == 0
    assert len(server.requests) == 3
    _assert_nothing_named(raised.value)


def test_a_body_that_is_not_utf8_is_a_response_error(server):
    server.script["/v13/deployments/dpl_x"] = [(200, '{"id": "é"}'.encode("latin-1"), {})]

    with pytest.raises(VercelResponseError, match="without a JSON body") as raised:
        _client().request(DEPLOYMENT_ROUTE, ids={"deployment": "dpl_x"})

    _assert_nothing_named(raised.value)


def test_a_project_synced_over_the_wire_lands_in_staging_with_complete_stats(
    server, monkeypatch, tmp_path
):
    import dlt

    fake = FakeVercel(team_id=TEAM)
    fake.add_project(PROJECT, "web")
    fake.add_deployment("dpl_ready", PROJECT, minutes_ago=30)
    fake.add_deployment("dpl_fail", PROJECT, state="ERROR", minutes_ago=10, errorCode="X")
    fake.set_log("dpl_fail", "Error: boom")
    server.fake = fake
    lines: list[str] = []

    def record(message, *args, **_kwargs):
        lines.append(message % args if args else message)

    monkeypatch.setattr(vercel_module, "logger", SimpleNamespace(warning=record, info=record))
    source = vercel_source(
        resource_name="vercel_wire",
        project_id=PROJECT,
        team_id=TEAM,
        access_token=TOKEN,
        include_build_logs=True,
    )
    pipeline = dlt.pipeline(
        pipeline_name="vercel_wire",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'w.db'}"),
        dataset_name="vercel",
        pipelines_dir=str(tmp_path / "pipelines"),
    )

    pipeline.run(source, columns=dlt_utils.DOCUMENT_COLUMN_HINTS)

    with (
        pipeline.sql_client() as sql,
        sql.execute_query('SELECT id FROM "vercel_wire" ORDER BY id') as cursor,
    ):
        staged = [row[0] for row in cursor.fetchall()]
    assert staged == [
        "build_log:dpl_fail",
        "deployment:dpl_fail",
        "deployment:dpl_ready",
        f"project:{PROJECT}",
    ]
    assert source.cognee_sync_stats == {**new_stats(), "scanned": 4}
    assert set(server.paths()) == {
        f"/v9/projects/{PROJECT}",
        "/v7/deployments",
        "/v3/deployments/dpl_fail/events",
    }
    assert all(r["headers"]["authorization"] == f"Bearer {TOKEN}" for r in server.requests)
    assert all(r["query"]["teamId"] == [TEAM] for r in server.requests)
    for line in lines:
        for secret in SECRETS:
            assert secret not in line, line


def test_no_log_record_of_any_logger_carries_the_token(server, caplog):
    """The client logs nothing itself. urllib3 logs each request line at DEBUG (path and query,
    so the team and project ids), which vercel.md says; the token is in no record."""
    import logging

    server.script["/v13/deployments/dpl_x"] = [_json(503, {}), _json(404, {})]

    with caplog.at_level(logging.DEBUG), pytest.raises(VercelNotFoundError, match="HTTP 404"):
        _client().request(DEPLOYMENT_ROUTE, ids={"deployment": "dpl_x"}, params={"teamId": TEAM})

    assert caplog.records, "urllib3 logged the requests"
    for record in caplog.records:
        assert TOKEN not in record.getMessage()
        assert "Bearer" not in record.getMessage()
    assert all(
        record.name.startswith("urllib3")
        for record in caplog.records
        if TEAM in record.getMessage()
    ), "only urllib3's own DEBUG request line names the team"
