"""Small async client for Google OAuth and OpenID Connect endpoints."""

import asyncio
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
TIMEOUT = aiohttp.ClientTimeout(total=60)


class GoogleAuthError(RuntimeError):
    """A stable, log-safe Google OAuth failure."""

    def __init__(self, operation: str, code: str):
        self.code = code
        super().__init__(f"Google {operation} failed: {code}")


async def _token_request(operation: str, payload: dict[str, str]) -> dict[str, Any]:
    async with (
        aiohttp.ClientSession(timeout=TIMEOUT) as session,
        session.post(TOKEN_URL, data=payload) as response,
    ):
        status = response.status
        try:
            body: dict[str, Any] = await response.json()
        except (aiohttp.ClientError, ValueError, asyncio.TimeoutError):
            # Preserve a stable error without exposing response bodies or tokens.
            raise GoogleAuthError(operation, f"http_{status}") from None

    error = body.get("error")
    if error or status != 200:
        raise GoogleAuthError(operation, str(error) if error else f"http_{status}")
    if not body.get("access_token"):
        raise GoogleAuthError(operation, "no_access_token")
    return body


async def exchange_code(
    code: str, *, client_id: str, client_secret: str, redirect_uri: str
) -> dict[str, Any]:
    """Exchange an authorization code for Google tokens."""
    return await _token_request(
        "code exchange",
        {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
    )


async def refresh_access_token(
    refresh_token: str, *, client_id: str, client_secret: str
) -> dict[str, Any]:
    """Refresh an access token while preserving the durable refresh token."""
    return await _token_request(
        "token refresh",
        {
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
        },
    )


async def revoke_token(token: str) -> None:
    """Best-effort revoke of the grant represented by ``token``."""
    async with (
        aiohttp.ClientSession(timeout=TIMEOUT) as session,
        session.post(REVOKE_URL, data={"token": token}) as response,
    ):
        if response.status != 200:
            logger.warning("Google token revoke returned HTTP %s", response.status)


async def _authorized_get(
    operation: str,
    url: str,
    access_token: str,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with (
        aiohttp.ClientSession(timeout=TIMEOUT) as session,
        session.get(
            url, params=params, headers={"Authorization": f"Bearer {access_token}"}
        ) as response,
    ):
        if response.status != 200:
            raise RuntimeError(f"Google {operation} failed: HTTP {response.status}")
        return await response.json()


async def fetch_userinfo(access_token: str) -> dict[str, Any]:
    """Resolve the stable Google subject and display identity."""
    return await _authorized_get("userinfo", USERINFO_URL, access_token)
