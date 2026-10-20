"""serve() records the Cloud workspace id so telemetry can join installs to tenants.

Cognee Cloud answers ``/api/v1/users/me`` with ``active_workspace_id`` (the
control-plane ``tenants.id``); a self-hosted server answers with fastapi-users'
``UserRead``, which has none. The saved connection must carry the id for Cloud
and nothing for self-hosted, and the lookup must never fail a connection.
"""

import uuid

import pytest
from aiohttp import web

from cognee.api.v1.serve import credentials, state
from cognee.api.v1.serve.serve import _serve_direct, _serve_saved


async def _start_instance(users_me):
    """A minimal instance: /health, /api/v1/datasets and a /users/me handler."""
    app = web.Application()
    app.router.add_get("/health", lambda request: web.Response(text="ok"))
    app.router.add_get("/api/v1/datasets", lambda request: web.json_response([]))
    app.router.add_get("/api/v1/users/me", users_me)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


@pytest.fixture
def credentials_file(monkeypatch, tmp_path):
    path = tmp_path / "cloud_credentials.json"
    monkeypatch.setattr(credentials, "_CREDENTIALS_FILE", path)
    yield path
    state.set_remote_client(None)


async def _connect(users_me):
    runner, url = await _start_instance(users_me)
    try:
        client = await _serve_direct(url, "ck_test")
        await client.close()
    finally:
        await runner.cleanup()
    return credentials.load_credentials()


@pytest.mark.asyncio
async def test_cloud_connection_saves_the_workspace_id(credentials_file):
    workspace_id = str(uuid.uuid4())

    async def users_me(request):
        return web.json_response({"id": str(uuid.uuid4()), "active_workspace_id": workspace_id})

    creds = await _connect(users_me)

    assert creds.tenant_id == workspace_id


@pytest.mark.asyncio
async def test_self_hosted_connection_saves_no_workspace(credentials_file):
    async def users_me(request):
        return web.json_response({"id": str(uuid.uuid4()), "email": "a@example.com"})

    creds = await _connect(users_me)

    assert creds.tenant_id == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "users_me",
    [
        lambda request: web.Response(status=404),
        lambda request: web.Response(status=500),
        lambda request: web.Response(text="not json", content_type="text/plain"),
        lambda request: web.json_response({"active_workspace_id": "not-a-uuid"}),
        lambda request: web.json_response(["unexpected"]),
    ],
)
async def test_workspace_lookup_failure_never_fails_the_connection(credentials_file, users_me):
    async def handler(request):
        return users_me(request)

    creds = await _connect(handler)

    assert creds.service_url and creds.tenant_id == ""


@pytest.mark.asyncio
async def test_saved_connection_without_workspace_is_backfilled(credentials_file):
    workspace_id = str(uuid.uuid4())

    async def users_me(request):
        return web.json_response({"active_workspace_id": workspace_id})

    runner, url = await _start_instance(users_me)
    try:
        credentials.save_credentials(credentials.CloudCredentials(service_url=url, api_key="k"))
        client = await _serve_saved(credentials.load_credentials())
        await client.close()
    finally:
        await runner.cleanup()

    assert credentials.load_credentials().tenant_id == workspace_id
