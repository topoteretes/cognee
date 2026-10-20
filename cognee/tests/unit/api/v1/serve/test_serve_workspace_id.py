"""serve() records the Cloud workspace id so telemetry can join installs to tenants.

``active_workspace_id`` from ``/api/v1/users/me`` is the source of truth; until
Cloud tenant instances return it, the id comes from the instance URL
(``https://tenant-<id>.aws.cognee.ai``). Self-hosted servers have neither, so
nothing is recorded, and the lookup must never fail a connection.
"""

import uuid

import pytest
from aiohttp import web

from cognee.api.v1.serve import credentials, state
from cognee.api.v1.serve.cloud_client import CloudClient, _tenant_id_from_cloud_url
from cognee.api.v1.serve.serve import _serve_direct, _serve_saved

TENANT_ID = "f7974d52-a58b-4497-8c9c-5369fa413d41"


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
    """What an OSS server (and today's Cloud tenant instance) returns from /users/me."""
    return {
        "id": str(uuid.uuid4()),
        "email": "a@example.com",
        "is_active": True,
        "is_superuser": False,
        "is_verified": True,
        "parent_user_id": None,
        "tenant_id": tenant_id,
    }


@pytest.mark.parametrize(
    "url",
    [
        f"https://tenant-{TENANT_ID}.aws.cognee.ai",
        f"https://tenant-{TENANT_ID}.aws.cognee.ai/",
        f"https://{TENANT_ID}.aws.cognee.ai",
        f"https://TENANT-{TENANT_ID.upper()}.aws.cognee.ai",
    ],
)
def test_tenant_id_is_read_from_a_cloud_instance_url(url):
    assert _tenant_id_from_cloud_url(url) == TENANT_ID


@pytest.mark.parametrize(
    "url",
    [
        f"https://tenant-{TENANT_ID}.example.com",  # not Cloud, even with a UUID
        f"https://{TENANT_ID}.cognee.ai.example.com",
        "https://platform.cognee.ai",  # Cloud host without a tenant
        "http://localhost:8000",
        "https://[bad",
        "",
    ],
)
def test_no_tenant_id_from_other_urls(url):
    assert _tenant_id_from_cloud_url(url) == ""


@pytest.mark.asyncio
async def test_active_workspace_id_is_saved_from_any_host(credentials_file):
    workspace_id = str(uuid.uuid4())

    async def users_me(request):
        return web.json_response(
            {**_user_read(str(uuid.uuid4())), "active_workspace_id": workspace_id}
        )

    creds = await _connect(users_me)

    assert creds.tenant_id == workspace_id


@pytest.mark.asyncio
async def test_self_hosted_tenant_id_is_not_recorded(credentials_file):
    """A self-hosted server reports its local tenant_id; it is not a Cloud tenant."""

    async def users_me(request):
        return web.json_response(_user_read(str(uuid.uuid4())))

    creds = await _connect(users_me)

    assert creds.tenant_id == ""


@pytest.mark.asyncio
async def test_cloud_url_is_the_fallback_when_users_me_has_no_workspace(monkeypatch):
    client = CloudClient(f"https://tenant-{TENANT_ID}.aws.cognee.ai", "ck_x")

    async def unreachable():
        raise ConnectionError("no network in unit tests")

    monkeypatch.setattr(client, "_get_session", unreachable)

    assert await client._get_workspace_id() == TENANT_ID


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
        return web.json_response({**_user_read(None), "active_workspace_id": workspace_id})

    runner, url = await _start_instance(users_me)
    try:
        credentials.save_credentials(credentials.CloudCredentials(service_url=url, api_key="k"))
        client = await _serve_saved(credentials.load_credentials())
        await client.close()
    finally:
        await runner.cleanup()

    assert credentials.load_credentials().tenant_id == workspace_id
