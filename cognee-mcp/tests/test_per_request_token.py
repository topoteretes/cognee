"""Per-request API token resolution (multi-tenant API mode).

One MCP instance must serve N users: every outgoing cognee-API call has to
carry the *caller's* token, not the static --api-token the process was
started with. The caller's token is resolved per request, in priority order:

1. an explicit override set via ``set_request_token()`` (contextvar),
2. the ``Authorization: Bearer`` header of the HTTP request that carried the
   current MCP message (``request_ctx`` from the MCP SDK),
3. the static ``--api-token`` fallback (single-user behaviour, unchanged).
"""

import asyncio
import contextlib
import importlib
import sys
from pathlib import Path

import httpx
import pytest
import uvicorn
from starlette.requests import Request

from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

auth_context = importlib.import_module("src.auth_context")
CogneeClient = importlib.import_module("src.cognee_client").CogneeClient


def _http_request(headers: dict[str, str] | None = None) -> Request:
    """Build a starlette Request like the SDK attaches to request_ctx."""
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": raw_headers,
        "query_string": b"",
    }
    return Request(scope)


def _request_context(request: Request | None) -> RequestContext:
    """Minimal RequestContext as the lowlevel server sets it per message."""
    return RequestContext(
        request_id=1,
        meta=None,
        session=None,
        lifespan_context=None,
        request=request,
    )


# --- token resolution -------------------------------------------------------


def test_get_request_token_returns_none_outside_any_request():
    assert auth_context.get_request_token() is None


def test_get_request_token_reads_bearer_from_request_ctx():
    ctx_token = request_ctx.set(
        _request_context(_http_request({"Authorization": "Bearer caller-jwt"}))
    )
    try:
        assert auth_context.get_request_token() == "caller-jwt"
    finally:
        request_ctx.reset(ctx_token)


def test_get_request_token_ignores_non_bearer_authorization():
    ctx_token = request_ctx.set(
        _request_context(_http_request({"Authorization": "Basic dXNlcjpwdw=="}))
    )
    try:
        assert auth_context.get_request_token() is None
    finally:
        request_ctx.reset(ctx_token)


def test_get_request_token_handles_request_ctx_without_http_request():
    # stdio transport: request_ctx is set but carries no HTTP request
    ctx_token = request_ctx.set(_request_context(None))
    try:
        assert auth_context.get_request_token() is None
    finally:
        request_ctx.reset(ctx_token)


def test_explicit_override_wins_over_request_ctx():
    ctx_token = request_ctx.set(
        _request_context(_http_request({"Authorization": "Bearer from-header"}))
    )
    override = auth_context.set_request_token("explicit-token")
    try:
        assert auth_context.get_request_token() == "explicit-token"
    finally:
        auth_context.reset_request_token(override)
        request_ctx.reset(ctx_token)
    # after reset the header token is visible again
    try:
        assert auth_context.get_request_token() is None
    finally:
        pass


# --- CogneeClient header integration ----------------------------------------


def test_get_headers_prefers_request_token_over_static():
    client = CogneeClient(api_url="http://cognee.local", api_token="static-token")
    override = auth_context.set_request_token("per-request-jwt")
    try:
        headers = client._get_headers()
    finally:
        auth_context.reset_request_token(override)

    assert headers["Authorization"] == "Bearer per-request-jwt"


def test_get_headers_falls_back_to_static_token():
    client = CogneeClient(api_url="http://cognee.local", api_token="static-token")

    assert client._get_headers()["Authorization"] == "Bearer static-token"


def test_get_headers_without_any_token_has_no_authorization():
    client = CogneeClient(api_url="http://cognee.local")

    assert "Authorization" not in client._get_headers()


@pytest.mark.asyncio
async def test_api_calls_send_per_request_token_per_caller():
    """Acceptance shape: one client instance, two callers, two identities."""
    seen: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"status": "ok"})

    client = CogneeClient(api_url="http://cognee.local", api_token="static-token")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        for jwt in ("user-a-jwt", "user-b-jwt"):
            ctx_token = request_ctx.set(
                _request_context(_http_request({"Authorization": f"Bearer {jwt}"}))
            )
            try:
                await client.add("memory", dataset_name="ds")
            finally:
                request_ctx.reset(ctx_token)
        # no caller context -> static fallback
        await client.add("memory", dataset_name="ds")
    finally:
        await client.close()

    assert seen == ["Bearer user-a-jwt", "Bearer user-b-jwt", "Bearer static-token"]


@pytest.mark.asyncio
async def test_request_tokens_are_isolated_across_concurrent_tasks():
    client = CogneeClient(api_url="http://cognee.local", api_token="static-token")
    results: dict[str, str] = {}

    async def caller(name: str, jwt: str, delay: float) -> None:
        ctx_token = request_ctx.set(
            _request_context(_http_request({"Authorization": f"Bearer {jwt}"}))
        )
        try:
            await asyncio.sleep(delay)  # force interleaving
            results[name] = client._get_headers()["Authorization"]
        finally:
            request_ctx.reset(ctx_token)

    await asyncio.gather(caller("a", "token-a", 0.02), caller("b", "token-b", 0))

    assert results == {"a": "Bearer token-a", "b": "Bearer token-b"}


# --- full-stack integration ---------------------------------------------------


@pytest.mark.asyncio
async def test_streamable_http_stack_forwards_caller_token(monkeypatch):
    """The risky link: tool handlers run in the MCP session task, not in the
    ASGI request task. Prove the Authorization header of the HTTP call reaches
    the outgoing cognee-API request through the real streamable-HTTP stack."""
    import src.server as server
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    upstream_auth: list[str | None] = []

    async def cognee_api(request: httpx.Request) -> httpx.Response:
        upstream_auth.append(request.headers.get("authorization"))
        return httpx.Response(200, json=[])

    api_client = CogneeClient(api_url="http://cognee.local", api_token="static-token")
    await api_client.client.aclose()
    api_client.client = httpx.AsyncClient(transport=httpx.MockTransport(cognee_api))
    monkeypatch.setattr(server, "cognee_client", api_client)

    app = server.mcp.streamable_http_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    http_server = uvicorn.Server(config)
    serve_task = asyncio.create_task(http_server.serve())
    try:
        while not http_server.started:
            await asyncio.sleep(0.02)
        port = http_server.servers[0].sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/mcp"

        async def call_as(jwt: str) -> None:
            async with contextlib.AsyncExitStack() as stack:
                http_client = httpx.AsyncClient(headers={"Authorization": f"Bearer {jwt}"})
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(url, http_client=http_client)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                result = await session.call_tool("list_datasets_json", {})
                assert not result.isError

        await call_as("user-a-jwt")
        await call_as("user-b-jwt")
    finally:
        http_server.should_exit = True
        await serve_task
        await api_client.close()

    assert upstream_auth == ["Bearer user-a-jwt", "Bearer user-b-jwt"]
