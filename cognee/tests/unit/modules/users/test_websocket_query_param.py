"""resolve_websocket_query_param_fallback: the ?token= WS-only fallback.

A browser cannot set a custom header when opening a WebSocket, so a
header-based scheme (API key, bearer) has no way to see a credential on a
WS handshake unless it also accepts one via a query parameter. These pin
that the fallback only ever applies to a WebSocket connection, and only
when the header-based scheme found nothing.
"""

from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocket

from cognee.modules.users.authentication.websocket_query_param import (
    WEBSOCKET_QUERY_PARAM_NAME,
    resolve_websocket_query_param_fallback,
)


def _websocket(query_string: bytes = b"") -> WebSocket:
    scope = {
        "type": "websocket",
        "query_string": query_string,
        "headers": [],
        "path": "/ws",
    }
    return WebSocket(scope, receive=None, send=None)


@pytest.mark.asyncio
async def test_primary_token_wins_over_the_query_param():
    ws = _websocket(f"{WEBSOCKET_QUERY_PARAM_NAME}=from-query".encode())
    assert await resolve_websocket_query_param_fallback(ws, "from-header") == "from-header"


@pytest.mark.asyncio
async def test_websocket_falls_back_to_the_query_param_when_no_header_token():
    ws = _websocket(f"{WEBSOCKET_QUERY_PARAM_NAME}=from-query".encode())
    assert await resolve_websocket_query_param_fallback(ws, None) == "from-query"


@pytest.mark.asyncio
async def test_websocket_with_no_query_param_and_no_header_token_yields_none():
    ws = _websocket(b"")
    assert await resolve_websocket_query_param_fallback(ws, None) is None


@pytest.mark.asyncio
async def test_http_request_never_consults_the_query_param():
    # A plain object is enough here: the fallback only special-cases
    # `isinstance(request, WebSocket)`, so any non-WebSocket connection -
    # including a real Request - must fall through to None regardless of
    # what it carries, so a credential can never leak into an HTTP URL.
    http_request = SimpleNamespace(
        query_params={WEBSOCKET_QUERY_PARAM_NAME: "should-never-be-read"}
    )
    assert await resolve_websocket_query_param_fallback(http_request, None) is None
