"""A redirect from GitHub must not carry the client secret or a token anywhere.

aiohttp follows a 307/308 with the body by default, so these run against real servers
on loopback: it is aiohttp's behaviour being checked, not a keyword argument.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web

from cognee_connector_github import app_auth
from cognee_connector_github.cognee_host import adapter as adapter_module
from cognee_connector_github.cognee_host import sync as sync_module


@asynccontextmanager
async def _serve(app: web.Application):
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def _redirecting_api():
    """A server that 307s every path to a collector; yields its URL and the collector's log."""
    collected: list[str] = []

    async def collect(request: web.Request) -> web.Response:
        collected.append(request.path)
        return web.json_response({"access_token": "x", "token": "y", "installations": []})

    collector = web.Application()
    collector.router.add_route("*", "/collect", collect)
    async with _serve(collector) as collector_url:

        async def redirect(_request: web.Request) -> web.Response:
            raise web.HTTPTemporaryRedirect(location=f"{collector_url}/collect")

        api = web.Application()
        api.router.add_route("*", "/{tail:.*}", redirect)
        async with _serve(api) as api_url:
            yield api_url, collected


@pytest.mark.asyncio
async def test_the_user_token_exchange_does_not_follow_a_redirect():
    async with _redirecting_api() as (url, collected):
        with (
            patch.object(adapter_module, "_ACCESS_TOKEN_URL", f"{url}/token"),
            patch.object(adapter_module, "require", lambda name: f"test-{name}"),
            pytest.raises(RuntimeError, match="HTTP 307"),
        ):
            await adapter_module.GithubIntegration().exchange_code("the-code")
    assert collected == []


@pytest.mark.parametrize(
    "call",
    [
        lambda: app_auth.get_installation(42),
        lambda: app_auth.mint_installation_token(42),
        lambda: app_auth.user_can_access_installation("token", 42),
        lambda: app_auth.list_installation_repository_records("token"),
    ],
    ids=["get_installation", "mint_token", "user_installations", "repository_records"],
)
@pytest.mark.asyncio
async def test_the_app_auth_calls_do_not_follow_a_redirect(call):
    async with _redirecting_api() as (url, collected):
        with (
            patch.object(app_auth, "API_BASE_URL", url),
            patch.object(app_auth, "build_app_jwt", return_value="the-jwt"),
            pytest.raises(RuntimeError, match="HTTP 307"),
        ):
            await call()
    assert collected == []


@pytest.mark.asyncio
async def test_the_sync_repository_listing_does_not_follow_a_redirect():
    async with _redirecting_api() as (url, collected):
        with (
            patch.object(sync_module, "API_BASE_URL", url),
            patch.object(sync_module, "report_sync_progress", new=AsyncMock()),
            pytest.raises(RuntimeError, match="HTTP 307"),
        ):
            await sync_module.list_installation_repositories("token")
    assert collected == []
