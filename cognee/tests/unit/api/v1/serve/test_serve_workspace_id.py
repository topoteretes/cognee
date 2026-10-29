"""serve() records the Cloud workspace id so telemetry can join installs to tenants.

Every instance answers ``/api/v1/users/me`` with fastapi-users' ``UserRead``.
On a Cloud tenant instance its ``tenant_id`` is the control-plane
``tenants.id``; a self-hosted server reports its own local tenant. Both are
kept on the connected client for telemetry and saved for reconnects (analytics
tells them apart by host). The lookup must never fail or stall a connection.
"""

import asyncio
import time
import uuid

import pytest
from aiohttp import web

from cognee.api.v1.serve import credentials, state
from cognee.api.v1.serve.cloud_client import CloudClient
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


def _user_read(tenant_id):
    """The fields every instance returns from /api/v1/users/me."""
    return {
        "id": str(uuid.uuid4()),
        "email": "a@example.com",
        "is_active": True,
        "is_superuser": False,
        "is_verified": True,
        "parent_user_id": None,
        "tenant_id": tenant_id,
    }


@pytest.mark.asyncio
async def test_connection_saves_the_reported_tenant_id(credentials_file):
    tenant_id = str(uuid.uuid4())

    async def users_me(request):
        return web.json_response(_user_read(tenant_id))

    creds = await _connect(users_me)

    assert creds.tenant_id == tenant_id
    assert state.peek_remote_client().tenant_id == tenant_id


@pytest.mark.asyncio
async def test_user_without_tenant_saves_no_tenant(credentials_file):
    async def users_me(request):
        return web.json_response(_user_read(None))

    creds = await _connect(users_me)

    assert creds.tenant_id == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "users_me",
    [
        lambda request: web.Response(status=404),
        lambda request: web.Response(status=500),
        lambda request: web.Response(text="not json", content_type="text/plain"),
        lambda request: web.json_response({"tenant_id": "not-a-uuid"}),
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
        return web.json_response(_user_read(workspace_id))

    runner, url = await _start_instance(users_me)
    try:
        credentials.save_credentials(credentials.CloudCredentials(service_url=url, api_key="k"))
        client = await _serve_saved(credentials.load_credentials())
        await client.close()
    finally:
        await runner.cleanup()

    assert credentials.load_credentials().tenant_id == workspace_id


@pytest.mark.asyncio
async def test_saved_connection_keeps_its_tenant_on_the_client(credentials_file):
    tenant_id = str(uuid.uuid4())
    calls = []

    async def users_me(request):
        calls.append(request)
        return web.json_response(_user_read(str(uuid.uuid4())))

    runner, url = await _start_instance(users_me)
    try:
        credentials.save_credentials(
            credentials.CloudCredentials(service_url=url, api_key="k", tenant_id=tenant_id)
        )
        client = await _serve_saved(credentials.load_credentials())
        await client.close()
    finally:
        await runner.cleanup()

    assert client.tenant_id == tenant_id
    assert calls == []  # a saved tenant needs no lookup


@pytest.mark.asyncio
async def test_failed_backfill_save_keeps_the_working_connection(credentials_file, monkeypatch):
    tenant_id = str(uuid.uuid4())

    async def users_me(request):
        return web.json_response(_user_read(tenant_id))

    runner, url = await _start_instance(users_me)
    try:
        credentials.save_credentials(credentials.CloudCredentials(service_url=url, api_key="k"))
        saved = credentials.load_credentials()

        def read_only(creds):
            raise PermissionError("read-only credentials file")

        monkeypatch.setattr(credentials, "save_credentials", read_only)
        client = await _serve_saved(saved)
        await client.close()
    finally:
        await runner.cleanup()

    assert state.peek_remote_client() is client
    assert client.tenant_id == tenant_id


@pytest.mark.asyncio
async def test_stalled_tenant_lookup_does_not_delay_the_connection(credentials_file, monkeypatch):
    import aiohttp

    monkeypatch.setattr(CloudClient, "TENANT_LOOKUP_TIMEOUT", aiohttp.ClientTimeout(total=0.2))

    async def users_me(request):
        await asyncio.sleep(2)
        return web.json_response(_user_read(str(uuid.uuid4())))

    runner, url = await _start_instance(users_me)
    try:
        started = time.monotonic()
        client = await _serve_direct(url, "ck_test")
        elapsed = time.monotonic() - started
        await client.close()
    finally:
        await runner.cleanup()

    assert elapsed < 3
    assert client.tenant_id == ""
    assert state.peek_remote_client() is client
