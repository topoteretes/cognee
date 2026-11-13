"""The OAuth, account and folder-picker client: its error codes, one session per walk, and the
real wire against a local HTTP server.

Copied from the client tests of cognee.tests.unit.modules.integrations.test_dropbox_integration
(cognee dev, 2026-11-12); the adapter and sync tests there stay with cognee's integration module.
The session the picker reuses is ``aiohttp.ClientSession`` here, where cognee patches its
``client_session`` wrapper.
"""

import json

import pytest

from cognee_connector_dropbox import client as client_module


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"error": "invalid_grant", "error_description": "secret text"}, 400, "invalid_grant"),
        ({"error": {".tag": "path", "path": {".tag": "not_found"}}}, 409, "path"),
        (None, 500, "http_500"),
        ({"error": "Some Free Text"}, 400, "http_400"),
    ],
)
def test_client_errors_carry_a_tag_or_a_status_only(body, status, code):
    assert client_module._code(body, status) == code


@pytest.mark.asyncio
async def test_the_folder_picker_reuses_one_session_across_pages_and_keeps_folders_only(
    monkeypatch,
):
    pages = iter(
        [
            {
                "entries": [
                    {".tag": "folder", "id": "id:a", "path_display": "/A"},
                    {".tag": "file", "id": "id:f", "path_display": "/A/f.md"},
                ],
                "cursor": "c1",
                "has_more": True,
            },
            {
                "entries": [{".tag": "folder", "id": "id:b", "path_display": "/B"}],
                "cursor": "c2",
                "has_more": False,
            },
        ]
    )
    sessions = []

    class Response:
        status = 200

        async def json(self, content_type=None):
            return next(pages)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class Session:
        def __init__(self, **_kwargs):
            sessions.append(self)

        def post(self, *_args, **_kwargs):
            return Response()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setattr(client_module.aiohttp, "ClientSession", Session)

    folders = await client_module.list_folders("sl.token")

    assert [f["id"] for f in folders] == ["id:a", "id:b"]
    assert len(sessions) == 1, f"{len(sessions)} HTTP sessions for two pages"


@pytest.mark.asyncio
async def test_the_folder_listing_is_one_level_and_folders_only(monkeypatch):
    bodies = []

    async def post(operation, url, **kwargs):
        bodies.append(kwargs.get("json"))
        return {
            "entries": [
                {".tag": "folder", "id": "id:c", "path_display": "/A/c"},
                {".tag": "file", "id": "id:f", "path_display": "/A/f.md"},
            ],
            "cursor": "c1",
            "has_more": False,
        }

    monkeypatch.setattr(client_module, "_post", post)

    folders = await client_module.list_folders("t", parent="id:a")

    assert [f["id"] for f in folders] == ["id:c"]
    assert bodies[0]["recursive"] is False and bodies[0]["path"] == "id:a"


class _DropboxOAuthServer:
    """A real HTTP server on a free local port that answers like Dropbox's OAuth and RPC routes."""

    def __init__(self, answers):
        import http.server
        import threading

        self.answers = answers
        self.requests = []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                server.requests.append((self.path, dict(self.headers), body))
                status, payload = server.answers[self.path](body)
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"

    def close(self):
        self.http.shutdown()
        self.http.server_close()


@pytest.fixture
def oauth_server(monkeypatch):
    answers: dict = {}
    server = _DropboxOAuthServer(answers)
    for name, path in {
        "TOKEN_URL": "/oauth2/token",
        "REVOKE_URL": "/2/auth/token/revoke",
        "ACCOUNT_URL": "/2/users/get_current_account",
        "METADATA_URL": "/2/files/get_metadata",
        "LIST_URL": "/2/files/list_folder",
        "CONTINUE_URL": "/2/files/list_folder/continue",
    }.items():
        monkeypatch.setattr(client_module, name, server.base + path)
    server.answers_for = answers
    yield server
    server.close()


@pytest.mark.asyncio
async def test_the_token_exchange_and_refresh_cross_a_real_socket(oauth_server):
    oauth_server.answers_for["/oauth2/token"] = lambda body: (
        200,
        {"access_token": "sl.a", "account_id": "dbid:A", "expires_in": 14400},
    )

    granted = await client_module.exchange_code(
        "code", client_id="key", client_secret="secret", redirect_uri="https://x/cb"
    )
    renewed = await client_module.refresh_access_token(
        "refresh", client_id="key", client_secret="secret"
    )

    assert granted["account_id"] == "dbid:A" and renewed["access_token"] == "sl.a"
    (_path, headers, body), second = oauth_server.requests
    assert b"grant_type=authorization_code" in body and headers["Authorization"].startswith(
        "Basic "
    )
    assert b"grant_type=refresh_token" in second[2]


@pytest.mark.asyncio
async def test_get_current_account_crosses_a_real_socket(oauth_server):
    oauth_server.answers_for["/2/users/get_current_account"] = lambda body: (
        200,
        {"account_id": "dbid:A", "email": "a@example.com"},
    )

    account = await client_module.get_current_account("sl.t")

    assert account["email"] == "a@example.com"
    assert oauth_server.requests[0][1]["Authorization"] == "Bearer sl.t"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (200, {".tag": "folder", "id": "id:a", "path_lower": "/a"}, "folder"),
        (200, {".tag": "file", "id": "id:f"}, None),
        (409, {"error": {".tag": "path", "path": {".tag": "not_found"}}}, None),
        (401, {"error": {".tag": "expired_access_token"}}, "expired_access_token"),
        (429, {"error": {".tag": "too_many_requests"}}, "too_many_requests"),
    ],
)
async def test_get_folder_crosses_a_real_socket(oauth_server, status, payload, expected):
    oauth_server.answers_for["/2/files/get_metadata"] = lambda body: (status, payload)

    if status in (401, 429):
        with pytest.raises(client_module.DropboxIntegrationError) as error:
            await client_module.get_folder("sl.t", "id:a")
        assert error.value.code == expected and "sl.t" not in str(error.value)
        return
    folder = await client_module.get_folder("sl.t", "id:a")
    assert (folder or {}).get(".tag") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("parent", [None, "id:a"])
async def test_list_folders_crosses_a_real_socket_with_a_continue_page(oauth_server, parent):
    oauth_server.answers_for["/2/files/list_folder"] = lambda body: (
        200,
        {"entries": [{".tag": "folder", "id": "id:1"}], "cursor": "c1", "has_more": True},
    )
    oauth_server.answers_for["/2/files/list_folder/continue"] = lambda body: (
        200,
        {
            "entries": [{".tag": "folder", "id": "id:2"}, {".tag": "file", "id": "id:f"}],
            "cursor": "c2",
            "has_more": False,
        },
    )

    folders = await client_module.list_folders("sl.t", parent=parent)

    assert [f["id"] for f in folders] == ["id:1", "id:2"]
    first = json.loads(oauth_server.requests[0][2])
    assert first == {"path": parent or "", "recursive": False, "limit": 2000}
    assert json.loads(oauth_server.requests[1][2]) == {"cursor": "c1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"null", b""])
async def test_revoke_crosses_a_real_socket_and_a_void_answer_is_success(oauth_server, payload):
    """auth/token/revoke is a void RPC: Dropbox answers 200 with a null (or empty) body."""
    oauth_server.answers_for["/2/auth/token/revoke"] = lambda body: (200, payload)

    await client_module.revoke_token("sl.t")

    assert oauth_server.requests[0][1]["Authorization"] == "Bearer sl.t"
