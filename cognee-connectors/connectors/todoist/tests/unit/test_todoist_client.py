"""The Todoist API client: request shape, error mapping, retries and token hygiene.

Importing the connectors loads no HTTP client library, so ``httpx`` is imported when the
first request is made. The token never reaches an error message, a ``__cause__``, a repr
or a log line.
"""

import json
import logging
import os
import subprocess
import sys
import textwrap

import httpx
import pytest
from fake_todoist import FakeTodoist

from cognee_connector_todoist.client import (
    REST_BASE_URL,
    SYNC_URL,
    TodoistAPIError,
    TodoistAuthError,
    TodoistClient,
    TodoistRateLimitedError,
    TodoistResponseError,
    build_todoist_service,
)

TOKEN = "SECRET-todoist-token-123"

# A `None` entry makes any later `import httpx` raise: the import below only succeeds
# when no module of the package asks for it at import time (the package needs no host to
# import, so nothing loads httpx before it).
_IMPORT_WITHOUT_HTTPX = textwrap.dedent(
    """
    import sys

    # A None entry makes any later `import httpx` raise. Reaching the last line proves
    # no module of the package asked for httpx at import time.
    sys.modules["httpx"] = None

    import cognee_connector_todoist
    import cognee_connector_todoist.client as client
    import cognee_connector_todoist.source as source

    assert not hasattr(client, "httpx")
    assert not hasattr(source, "httpx")
    """
)


class Resp:
    """The part of ``httpx.Response`` the client reads."""

    def __init__(self, status: int, payload, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _client(monkeypatch, *responses):
    """A real client whose transport answers from ``responses`` in order.

    The retry sleeper records into ``sleeps`` instead of sleeping, so retry budgets are
    tested without waiting.
    """
    calls, sleeps = [], []
    queue = list(responses)

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("httpx.request", fake_request)
    return TodoistClient(TOKEN, sleep=sleeps.append), calls, sleeps


def test_importing_the_connector_does_not_load_httpx():
    # The package is found the way this process found it: installed, or on PYTHONPATH.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_HTTPX],
        check=False,
        timeout=120,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "token",
    ["", "   ", "bad token", "bäd", "\x00", None, 42],
    ids=["empty", "blank", "inner-whitespace", "non-ascii", "control", "none", "not-a-string"],
)
def test_a_malformed_token_is_refused_without_being_echoed(token):
    with pytest.raises(ValueError) as raised:
        TodoistClient(token)  # type: ignore[arg-type]

    assert str(raised.value) == "The Todoist API token is empty or has an invalid format"
    assert TOKEN not in str(raised.value)


def test_the_repr_is_redacted():
    assert "SECRET" not in repr(TodoistClient(TOKEN))


def test_a_request_sends_the_bearer_token_the_params_and_the_form(monkeypatch):
    client, calls, _ = _client(monkeypatch, Resp(200, {"ok": True}))

    assert client.request(
        "POST", SYNC_URL, data={"sync_token": "*", "resource_types": '["items"]'}
    ) == {"ok": True}

    method, url, kwargs = calls[0]
    assert (method, url) == ("POST", SYNC_URL)
    assert kwargs["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert kwargs["data"] == {"sync_token": "*", "resource_types": '["items"]'}
    assert kwargs["timeout"]


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, TodoistAuthError),
        (403, TodoistAuthError),
        (404, TodoistAPIError),
        (400, TodoistAPIError),
    ],
)
def test_error_statuses_map_to_typed_errors(monkeypatch, status, error):
    client, _, _ = _client(monkeypatch, Resp(status, {"error": "No way"}))

    with pytest.raises(error) as raised:
        client.request("GET", "/tasks/1234567890")

    assert type(raised.value) is error
    assert raised.value.status == status
    assert str(raised.value) == f"Todoist /tasks/{{id}}: HTTP {status}"


def test_the_error_message_never_names_a_task_id_and_todoists_explanation_is_an_attribute(
    monkeypatch,
):
    client, _, _ = _client(monkeypatch, Resp(403, {"error": "Task 1234567890 is private"}))

    with pytest.raises(TodoistAPIError) as raised:
        client.request("GET", "/tasks/1234567890")

    assert str(raised.value) == "Todoist /tasks/{id}: HTTP 403"
    assert raised.value.endpoint == "/tasks/{id}"
    assert raised.value.detail == "Task 1234567890 is private"


def test_the_token_is_cut_out_of_todoists_explanation(monkeypatch):
    client, _, _ = _client(monkeypatch, Resp(401, {"error": f"Invalid token {TOKEN}"}))

    with pytest.raises(TodoistAuthError) as raised:
        client.request("GET", "/projects")

    assert raised.value.detail == "Invalid token [token]"
    assert TOKEN not in str(raised.value)


def test_a_429_is_retried_with_retry_after_and_then_raised(monkeypatch):
    client, _, sleeps = _client(
        monkeypatch,
        Resp(429, {"error": "Limit hit"}, headers={"retry-after": "1"}),
        Resp(429, {"error": "Limit hit"}, headers={"retry-after": "3"}),
        Resp(429, {"error": "Limit hit"}),
        Resp(429, {"error": "Limit hit"}),
        Resp(429, {"error": "Limit hit"}),
    )

    with pytest.raises(TodoistRateLimitedError):
        client.request("GET", "/comments")

    # The header is honored while present; a missing one falls back to backoff.
    assert sleeps == [1.0, 3.0, 4.0, 8.0]


def test_a_5xx_is_retried_with_backoff_and_a_404_is_not(monkeypatch):
    client, _, sleeps = _client(
        monkeypatch,
        Resp(500, {"error": "boom"}),
        Resp(200, {"ok": True}),
    )
    assert client.request("GET", "/projects") == {"ok": True}
    assert sleeps == [1.0]

    client, _, sleeps = _client(monkeypatch, Resp(404, {"error": "gone"}))
    with pytest.raises(TodoistAPIError):
        client.request("GET", "/tasks/1")
    assert sleeps == []


def test_a_transport_error_is_retried_and_then_raised_as_no_answer(monkeypatch):
    client, _, _ = _client(
        monkeypatch,
        httpx.ConnectError("nope"),
        httpx.ConnectError("nope"),
        httpx.ConnectError("nope"),
        httpx.ConnectError("nope"),
        httpx.ConnectError("nope"),
    )

    with pytest.raises(TodoistAPIError) as raised:
        client.request("GET", "/projects")

    assert raised.value.status == 0
    assert str(raised.value) == "Todoist /projects: no answer"


def test_sync_posts_the_cursor_and_the_resource_types(monkeypatch):
    client, calls, _ = _client(monkeypatch, Resp(200, {"sync_token": "7"}))

    answer = client.sync("*", ["items", "projects", "notes"])

    assert answer == {"sync_token": "7"}
    _, url, kwargs = calls[0]
    assert url == SYNC_URL
    body = kwargs["data"]
    assert body["sync_token"] == "*"
    assert json.loads(body["resource_types"]) == ["items", "projects", "notes"]


def test_list_comments_paginates_until_there_is_no_next_cursor(monkeypatch):
    client, calls, _ = _client(
        monkeypatch,
        Resp(200, {"results": [{"id": "c1"}], "next_cursor": "page-2"}),
        Resp(200, {"results": [{"id": "c2"}]}),
    )

    assert [c["id"] for c in client.list_comments("42")] == ["c1", "c2"]
    assert calls[1][2]["params"] == {"task_id": "42", "cursor": "page-2"}


def test_list_comments_refuses_a_body_of_the_wrong_shape(monkeypatch):
    client, _, _ = _client(monkeypatch, Resp(200, {"results": "not-a-list"}))

    with pytest.raises(TodoistResponseError, match="results is not a list"):
        client.list_comments("42")


def test_a_200_without_a_json_object_is_a_response_error(monkeypatch):
    client, _, _ = _client(monkeypatch, Resp(200, ["not", "an", "object"]))

    with pytest.raises(TodoistResponseError, match="HTTP 200 without a JSON object"):
        client.request("GET", "/projects")


def test_build_todoist_service_wraps_the_token():
    service = build_todoist_service(TOKEN)

    assert isinstance(service, TodoistClient)
    assert "SECRET" not in repr(service)


def test_the_fake_answers_only_what_the_source_calls():
    fake = FakeTodoist()
    fake.add_project("P1")
    fake.add_task("1", "P1")

    answer = fake.sync("*", ["items", "projects", "notes"])
    assert answer["full_sync"] is True
    assert [p["id"] for p in answer["projects"]] == ["P1"]
    assert [i["id"] for i in answer["items"]] == ["1"]

    # An incremental sync with the answered token sees nothing new.
    assert fake.sync(answer["sync_token"], ["items"])["items"] == []

    fake.edit_task("1", content="Renamed")
    changed = fake.sync(answer["sync_token"], ["items"])
    assert [i["content"] for i in changed["items"]] == ["Renamed"]
