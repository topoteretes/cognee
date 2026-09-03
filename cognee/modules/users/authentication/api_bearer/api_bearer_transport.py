from typing import Optional

from fastapi.security import OAuth2PasswordBearer
from starlette.requests import HTTPConnection
from fastapi_users.authentication import BearerTransport

from cognee.modules.users.authentication.websocket_query_param import (
    resolve_websocket_query_param_fallback,
)

_TOKEN_URL = "/api/v1/auth/token"


class _OAuth2PasswordBearerOrWebSocketQueryParam(OAuth2PasswordBearer):
    """``OAuth2PasswordBearer`` that also accepts the token via ``?token=`` on
    a WebSocket handshake.

    See ``resolve_websocket_query_param_fallback`` for why the fallback is
    scoped to WebSocket connections only.
    """

    def __init__(self, token_url: str):
        super().__init__(tokenUrl=token_url, auto_error=False)

    # HTTPConnection, not Request: FastAPI injects a parameter only when its
    # annotation is one of the connection types it recognises, and this scheme
    # has to serve both HTTP routes and WebSocket handshakes. Left unannotated,
    # FastAPI read it as an ordinary query parameter named `request` and every
    # unauthenticated HTTP call answered 422 (missing query param) instead of
    # 401. Guarded by tests/unit/modules/users/test_transport_scheme_signature.py.
    async def __call__(self, request: HTTPConnection) -> Optional[str]:
        token = await super().__call__(request)
        return await resolve_websocket_query_param_fallback(request, token)


api_bearer_transport = BearerTransport(
    tokenUrl=_TOKEN_URL,
)
api_bearer_transport.scheme = _OAuth2PasswordBearerOrWebSocketQueryParam(_TOKEN_URL)

api_bearer_transport.name = "bearer"
