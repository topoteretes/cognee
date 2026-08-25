from typing import Optional

import jwt
from fastapi import HTTPException, WebSocket
from fastapi_users import exceptions as fastapi_users_exceptions

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

from ..get_fastapi_users import get_fastapi_users
from ..get_user_db import get_user_db_context
from ..get_user_manager import get_user_manager_context
from ..models import User
from .get_authenticated_user import REQUIRE_AUTHENTICATION
from .get_default_user import get_default_user
from .get_user import get_user

logger = get_logger("get_authenticated_websocket_user")

# What a credential being wrong looks like, as opposed to the server being
# broken. The strategies shipped here already fold this into a None return —
# fastapi-users' JWTStrategy swallows PyJWTError/UserNotExists/InvalidID, and
# ApiKeyJWTStrategy returns None for an unknown key — so this only matters for
# a deployment that plugs in a strategy which signals a bad credential by
# raising. Everything else (a dead database, a broken identity provider) is a
# server fault and must not be dressed up as a rejected credential.
BAD_CREDENTIAL_ERRORS = (
    fastapi_users_exceptions.FastAPIUsersException,
    jwt.PyJWTError,
)


async def _read_handshake_user(websocket: WebSocket) -> Optional[User]:
    """Resolve a user from the credentials carried by the WebSocket handshake.

    Walks the configured authentication backends in registration order and
    returns the first active user one of them yields — the same order and the
    same "first backend to yield a user wins" rule fastapi-users applies to
    HTTP requests, so a WebSocket accepts exactly the credentials the HTTP API
    accepts (API key header, bearer header, auth cookie) and no others.

    Each backend's ``transport.scheme`` reads only cookies, headers or query
    parameters, all of which a handshake carries: a Starlette ``WebSocket`` and
    a ``Request`` are both ``HTTPConnection``s, which is why the schemes can be
    called with one here even though they annotate ``Request``.

    Only a bad credential is absorbed. Anything else a backend raises — the
    strategies here reach the database to resolve a token — propagates, so a
    database outage fails the handshake as the server fault it is instead of
    being reported to every client as a permanent "Unauthorized".

    Raises:
        Exception: whatever a backend raised that was not a bad credential. It
            is logged here first, since a propagating dependency leaves no
            other trace of which backend failed.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        async with get_user_db_context(session) as user_db:
            async with get_user_manager_context(user_db) as user_manager:
                for backend in get_fastapi_users().authenticator.backends:
                    try:
                        # auto_error=False on every scheme here means "no
                        # credential" is a None return; a custom transport that
                        # sets auto_error raises instead, which is the same
                        # answer.
                        token = await backend.transport.scheme(websocket)
                    except HTTPException:
                        continue

                    if not token:
                        continue

                    try:
                        user = await backend.get_strategy().read_token(token, user_manager)
                    except BAD_CREDENTIAL_ERRORS as error:
                        # A failed authentication attempt: try the next backend
                        # and let the caller reject the connection if none
                        # match.
                        logger.debug(
                            "WebSocket authentication via %s failed: %s", backend.name, error
                        )
                        continue
                    except Exception:
                        logger.warning(
                            "WebSocket authentication backend %s failed to read a token",
                            backend.name,
                            exc_info=True,
                        )
                        raise

                    if user is not None and user.is_active:
                        return user

    return None


async def get_authenticated_websocket_user(websocket: WebSocket) -> Optional[User]:
    """Authenticate a WebSocket connection. The counterpart of ``get_authenticated_user``.

    A WebSocket route cannot depend on ``get_authenticated_user``: every
    ``fastapi.security`` scheme behind it takes a ``Request``, which FastAPI
    never supplies for a WebSocket connection, so the dependency raises before
    it ever reads a credential. This resolves the same credentials off the
    handshake instead, with the same posture as the HTTP dependency:

    - ``REQUIRE_AUTHENTICATION=true``: an unauthenticated connection returns
      None, and the route is expected to close it.
    - ``REQUIRE_AUTHENTICATION=false``: an unauthenticated connection falls
      back to the default user, so single-user deployments work without
      credentials exactly as their HTTP routes do.

    Returning None rather than raising is deliberate. A close frame sent before
    the handshake is accepted reaches the client as an HTTP rejection with no
    close code, so a route that wants clients to see *why* they were rejected
    (1008, "do not retry this") has to accept the connection first and close it
    itself — which it can only do if this returns instead of raising.

    Declared as a dependency so a deployment with its own scheme can replace it
    through ``app.dependency_overrides[get_authenticated_websocket_user]``,
    the same override point ``get_authenticated_user`` offers HTTP routes.

    Returns:
        The authenticated user, or None when authentication is required and the
        handshake carried no usable credential.

    Raises:
        Exception: any server fault met while resolving the credential — a
            backend failing for a reason other than a bad credential, the
            user re-fetch, or the default-user fallback. Deliberately not
            converted into None: a dependency resolves before the route can
            accept the connection, so the handshake simply fails and the
            client retries, instead of the client being told 1008
            ("Unauthorized", terminal, do not retry) because a database
            blinked. The failure is logged before it propagates.
    """
    user = await _read_handshake_user(websocket)

    if user is None:
        if REQUIRE_AUTHENTICATION:
            return None
        return await get_default_user()

    # Re-fetch for the same reason get_authenticated_user does: the instance a
    # strategy returns has no eagerly loaded relationships, and a long-lived
    # socket outlives the session it was read in.
    return await get_user(user.id)
