"""A redirect from a Slack endpoint must not carry the client secret or a token anywhere.

aiohttp follows a 307/308 with the body by default, so these run against real servers
on loopback: it is aiohttp's behaviour being checked, not a keyword argument.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web

from cognee_connector_slack.cognee_host import adapter as adapter_module
from cognee_connector_slack.cognee_host import oauth as oauth_module


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
async def _redirecting_endpoint():
    """An endpoint that 307s to a collector; yields its URL and what the collector got."""
    collected: list[bytes] = []

    async def collect(request: web.Request) -> web.Response:
        collected.append(await request.read())
        return web.json_response({"ok": True, "access_token": "x", "refresh_token": "y"})

    collector = web.Application()
    collector.router.add_route("*", "/collect", collect)
    async with _serve(collector) as collector_url:

        async def redirect(_request: web.Request) -> web.Response:
            raise web.HTTPTemporaryRedirect(location=f"{collector_url}/collect")

        endpoint = web.Application()
        endpoint.router.add_route("*", "/endpoint", redirect)
        async with _serve(endpoint) as endpoint_url:
            yield f"{endpoint_url}/endpoint", collected


def _require(name: str) -> str:
    return f"test-{name}"


@pytest.mark.asyncio
async def test_the_code_exchange_does_not_follow_a_redirect():
    async with _redirecting_endpoint() as (url, collected):
        with (
            patch.object(oauth_module, "_ACCESS_URL", url),
            patch.object(oauth_module, "require", _require),
            pytest.raises(RuntimeError, match="HTTP 307"),
        ):
            await oauth_module.exchange_code("the-code")
    assert collected == []


@pytest.mark.asyncio
async def test_the_token_refresh_does_not_follow_a_redirect():
    async with _redirecting_endpoint() as (url, collected):
        with (
            patch.object(oauth_module, "_ACCESS_URL", url),
            patch.object(adapter_module, "require", _require),
            patch.object(
                adapter_module, "decrypt_token_payload", return_value={"refresh_token": "r"}
            ),
            pytest.raises(RuntimeError, match="HTTP 307"),
        ):
            await adapter_module.SlackIntegration().refresh(SimpleNamespace())
    assert collected == []


@pytest.mark.asyncio
async def test_the_revoke_does_not_follow_a_redirect():
    async with _redirecting_endpoint() as (url, collected):
        with (
            patch.object(adapter_module, "_REVOKE_URL", url),
            patch.object(
                adapter_module, "decrypt_token_payload", return_value={"access_token": "a"}
            ),
        ):
            # A failed revoke is logged, never raised: it must not block a disconnect.
            await adapter_module.SlackIntegration().revoke_remote(
                SimpleNamespace(provider_account_id="T1")
            )
    assert collected == []
