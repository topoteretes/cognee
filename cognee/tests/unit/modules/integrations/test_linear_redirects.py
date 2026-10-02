"""Linear's OAuth POSTs never follow a redirect: the body carries secrets.

A 307 or 308 makes aiohttp send the same body to the host it names, and for the
refresh it would accept that host's answer as the new tokens.
"""

import pytest
import pytest_asyncio
from aiohttp import web

from cognee.modules.integrations.linear import adapter
from cognee.modules.integrations.linear.adapter import LinearAuthError, LinearIntegration


async def _serve(handler):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


@pytest_asyncio.fixture
async def redirecting_token_endpoint(monkeypatch):
    """Linear's URL answers 307 to a second server, which records what it receives."""
    received = []

    async def elsewhere(request):
        received.append(await request.text())
        return web.json_response({"access_token": "stolen", "refresh_token": "x"})

    other, other_port = await _serve(elsewhere)

    async def linear(request):
        # A fixed target: nothing from the request decides where this redirects.
        raise web.HTTPTemporaryRedirect(f"http://localhost:{other_port}/elsewhere")

    origin, port = await _serve(linear)
    monkeypatch.setattr(adapter, "_TOKEN_URL", f"http://127.0.0.1:{port}/oauth/token")
    monkeypatch.setattr(adapter, "_REVOKE_URL", f"http://127.0.0.1:{port}/oauth/revoke")
    yield received
    await origin.cleanup()
    await other.cleanup()


@pytest.mark.asyncio
async def test_a_refresh_does_not_follow_a_redirect_with_the_client_secret(
    redirecting_token_endpoint,
):
    with pytest.raises(LinearAuthError) as raised:
        await adapter.refresh_access_token("refresh-0", client_id="id", client_secret="secret")

    assert raised.value.code == "http_307"
    assert redirecting_token_endpoint == []


@pytest.mark.asyncio
async def test_a_revoke_does_not_follow_a_redirect_with_the_token(
    redirecting_token_endpoint, monkeypatch
):
    monkeypatch.setattr(
        adapter,
        "decrypt_token_payload",
        lambda credential: {"access_token": "access-0", "refresh_token": "refresh-0"},
    )
    credential = type("Credential", (), {"provider_account_id": "org-1"})()

    await LinearIntegration().revoke_remote(credential)

    assert redirecting_token_endpoint == []


@pytest.mark.asyncio
async def test_a_code_exchange_does_not_follow_a_redirect_with_the_client_secret(
    redirecting_token_endpoint, monkeypatch
):
    monkeypatch.setattr(adapter, "require", lambda key: "test")

    with pytest.raises(RuntimeError, match="HTTP 307"):
        await LinearIntegration().exchange_code("the-code")

    assert redirecting_token_endpoint == []
