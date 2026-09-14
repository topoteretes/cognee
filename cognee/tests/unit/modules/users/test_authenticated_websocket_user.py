"""get_authenticated_websocket_user: authenticating a handshake.

A WebSocket route cannot depend on get_authenticated_user — the security
schemes behind it require a Request FastAPI never supplies for a WebSocket —
so this is the seam both WS routes authenticate through instead. What it must
guarantee, and what these pin: the same credentials the HTTP API accepts are
accepted here in the same backend order, an inactive or unknown user does not
authenticate, a broken backend cannot take the others down with it, and the
REQUIRE_AUTHENTICATION posture behaves exactly as it does over HTTP.
"""

import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import jwt
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from cognee.modules.users.methods import get_authenticated_websocket_user

ws_auth_module = sys.modules["cognee.modules.users.methods.get_authenticated_websocket_user"]


def _user(is_active: bool = True):
    return SimpleNamespace(id=uuid4(), email="user@example.com", is_active=is_active)


def _backend(name: str, token, user=None, read_token=None):
    """A fastapi-users-shaped backend: a scheme that reads the handshake and a
    strategy that resolves whatever it found."""
    if read_token is None:

        async def read_token(_token, _user_manager):
            return user

    async def scheme(_connection):
        return token

    return SimpleNamespace(
        name=name,
        transport=SimpleNamespace(scheme=scheme),
        get_strategy=lambda: SimpleNamespace(read_token=read_token),
    )


@asynccontextmanager
async def _context(value=None):
    yield value


@pytest.fixture
def handshake(monkeypatch):
    """Patch away the user-manager plumbing; return a hook for the backends."""

    def _configure(backends, require_authentication=True):
        monkeypatch.setattr(
            ws_auth_module,
            "get_fastapi_users",
            lambda: SimpleNamespace(authenticator=SimpleNamespace(backends=backends)),
        )
        monkeypatch.setattr(
            ws_auth_module,
            "get_relational_engine",
            lambda: SimpleNamespace(get_async_session=_context),
        )
        monkeypatch.setattr(ws_auth_module, "get_user_db_context", lambda _session: _context())
        monkeypatch.setattr(ws_auth_module, "get_user_manager_context", lambda _db: _context())
        monkeypatch.setattr(ws_auth_module, "REQUIRE_AUTHENTICATION", require_authentication)
        # get_user re-fetches for eagerly loaded relationships; identity is all
        # these tests care about.
        monkeypatch.setattr(
            ws_auth_module, "get_user", AsyncMock(side_effect=lambda user_id: ("fetched", user_id))
        )

    return _configure


@pytest.mark.asyncio
async def test_the_first_backend_with_a_credential_wins(handshake):
    api_key_user, cookie_user = _user(), _user()
    handshake(
        [
            _backend("apikey", token="key-token", user=api_key_user),
            _backend("cookie", token="cookie-token", user=cookie_user),
        ]
    )

    resolved = await get_authenticated_websocket_user(SimpleNamespace())

    assert resolved == ("fetched", api_key_user.id)


@pytest.mark.asyncio
async def test_a_backend_without_a_credential_is_skipped_not_failed(handshake):
    cookie_user = _user()
    handshake(
        [
            _backend("apikey", token=None),
            _backend("cookie", token="cookie-token", user=cookie_user),
        ]
    )

    resolved = await get_authenticated_websocket_user(SimpleNamespace())

    assert resolved == ("fetched", cookie_user.id)


@pytest.mark.asyncio
async def test_a_bad_credential_does_not_stop_the_other_backends(handshake):
    """A malformed token is a failed attempt, not a server fault."""

    async def explode(_token, _user_manager):
        raise jwt.DecodeError("not a valid jwt")

    cookie_user = _user()
    handshake(
        [
            _backend("bearer", token="garbage", read_token=explode),
            _backend("cookie", token="cookie-token", user=cookie_user),
        ]
    )

    resolved = await get_authenticated_websocket_user(SimpleNamespace())

    assert resolved == ("fetched", cookie_user.id)


@pytest.mark.asyncio
async def test_a_server_fault_propagates_instead_of_reading_as_unauthorized(handshake, caplog):
    """The distinction this file exists to protect: a database that is down is
    not a client whose credential was rejected. Swallowing it would close every
    connection with 1008 — documented as terminal, do not retry — while the
    fault is transient, so clients would give up exactly when they should not.
    """

    async def database_is_down(_token, _user_manager):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    handshake(
        [
            _backend("apikey", token="valid-looking-key", read_token=database_is_down),
            _backend("cookie", token="cookie-token", user=_user()),
        ]
    )

    with pytest.raises(OperationalError):
        await get_authenticated_websocket_user(SimpleNamespace())

    # A propagating dependency leaves no other trace of which backend broke.
    assert "apikey" in caplog.text


@pytest.mark.asyncio
async def test_an_inactive_user_does_not_authenticate(handshake):
    """HTTP routes ask for active=True; a socket must not be a way around it."""
    handshake([_backend("cookie", token="cookie-token", user=_user(is_active=False))])

    assert await get_authenticated_websocket_user(SimpleNamespace()) is None


@pytest.mark.asyncio
async def test_an_unknown_token_returns_none_when_authentication_is_required(handshake):
    handshake([_backend("apikey", token="revoked-key", user=None)])

    assert await get_authenticated_websocket_user(SimpleNamespace()) is None


@pytest.mark.asyncio
async def test_no_credentials_falls_back_to_the_default_user_when_auth_is_off(
    handshake, monkeypatch
):
    """Single-user deployments reach their WebSockets exactly as their HTTP
    routes do — which the cookie-only path never allowed."""
    default_user = _user()
    handshake([_backend("cookie", token=None)], require_authentication=False)
    monkeypatch.setattr(ws_auth_module, "get_default_user", AsyncMock(return_value=default_user))

    assert await get_authenticated_websocket_user(SimpleNamespace()) is default_user


# The tests above hand-roll the backends, which proves the walk but assumes the
# thing most likely to break silently: that a fastapi-users security scheme,
# annotated for a Request, can in fact be called with a Starlette WebSocket.
# The tests below make no such assumption — real registered backends, a real
# APIKeyHeader scheme, a real handshake — so a fastapi or fastapi-users bump
# that ends the duck-typing fails here instead of degrading every socket in
# the product to 1008.

API_KEY = "cognee-test-api-key"


@pytest.fixture
def transport_app(monkeypatch):
    """A WebSocket route behind the real dependency and the real backends.

    Only the persistence under the backends is replaced: the user manager the
    api-key strategy resolves its token through, and the re-fetch afterwards.
    """
    key_holder = _user()

    async def get_by_token(token):
        return key_holder if token == API_KEY else None

    monkeypatch.setattr(
        ws_auth_module,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=_context),
    )
    monkeypatch.setattr(ws_auth_module, "get_user_db_context", lambda _session: _context())
    monkeypatch.setattr(
        ws_auth_module,
        "get_user_manager_context",
        lambda _db: _context(SimpleNamespace(get_by_token=get_by_token)),
    )
    monkeypatch.setattr(ws_auth_module, "REQUIRE_AUTHENTICATION", True)
    monkeypatch.setattr(ws_auth_module, "get_user", AsyncMock(return_value=key_holder))

    application = FastAPI()

    @application.websocket("/subscribe")
    async def subscribe(websocket: WebSocket):
        user = await get_authenticated_websocket_user(websocket)
        await websocket.accept()
        await websocket.send_json({"email": user.email if user else None})

    return application


def test_a_real_x_api_key_handshake_authenticates_through_the_real_schemes(transport_app):
    with TestClient(transport_app).websocket_connect(
        "/subscribe", headers={"X-Api-Key": API_KEY}
    ) as connection:
        assert connection.receive_json() == {"email": "user@example.com"}


def test_a_real_handshake_without_a_credential_authenticates_nobody(transport_app):
    with TestClient(transport_app).websocket_connect("/subscribe") as connection:
        assert connection.receive_json() == {"email": None}


def test_a_real_handshake_with_an_unknown_api_key_authenticates_nobody(transport_app):
    with TestClient(transport_app).websocket_connect(
        "/subscribe", headers={"X-Api-Key": "revoked"}
    ) as connection:
        assert connection.receive_json() == {"email": None}
