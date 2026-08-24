from typing import Optional

from starlette.requests import HTTPConnection
from starlette.websockets import WebSocket

# Shared across every transport's query-param fallback so a WS client only
# has to know one name regardless of which backend ends up matching it.
WEBSOCKET_QUERY_PARAM_NAME = "token"


async def resolve_websocket_query_param_fallback(
    request: HTTPConnection, primary_token: Optional[str]
) -> Optional[str]:
    """Fall back to a ``?token=`` query parameter when a header-based scheme found nothing.

    Browsers cannot set custom headers when opening a WebSocket, so a header
    the scheme normally requires is simply unavailable to a browser WS
    client — the query param is its only way to send the credential. Plain
    HTTP requests still require the header: the query param is consulted
    only when the connection is a WebSocket.

    This does NOT keep the credential out of HTTP-layer logs: a WebSocket
    handshake is itself an HTTP GET/Upgrade request, and its full URL
    (including the query string) is captured by the default access-log
    format of common reverse proxies and load balancers (e.g. nginx,
    AWS ALB), unlike an Authorization header, which those defaults do not
    log. Deployments that terminate WebSocket traffic through such a proxy
    should redact the ``token`` query parameter from its access logs.

    Uvicorn's own logs are covered regardless: importing
    ``cognee.api.client`` installs a filter (see
    ``redact_websocket_query_secrets.py``) that redacts this query param from
    uvicorn's access/error loggers, since uvicorn logs the full handshake
    path — proxy or not — on every WebSocket accept/close.
    """
    if primary_token:
        return primary_token
    if isinstance(request, WebSocket):
        return request.query_params.get(WEBSOCKET_QUERY_PARAM_NAME)
    return None
