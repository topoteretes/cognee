"""Auth0 Device Code Flow (RFC 8628) for CLI/SDK authentication."""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from cognee.shared.logging_utils import get_logger

logger = get_logger("serve.device_auth")

# Defaults — override via env vars
DEFAULT_AUTH0_DOMAIN = "cognee.eu.auth0.com"
DEFAULT_AUTH0_AUDIENCE = "cognee:api"
DEFAULT_SCOPE = "openid profile email offline_access"


@dataclass
class TokenResponse:
    access_token: str
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: int = 3600


def _get_auth0_domain() -> str:
    return os.getenv("COGNEE_AUTH0_DOMAIN", DEFAULT_AUTH0_DOMAIN)


def _get_auth0_client_id() -> str:
    client_id = os.getenv("COGNEE_AUTH0_DEVICE_CLIENT_ID", "")
    if not client_id:
        raise ValueError(
            "COGNEE_AUTH0_DEVICE_CLIENT_ID must be set. "
            "Create a 'Native App' in your Auth0 dashboard with Device Code grant enabled."
        )
    return client_id


def _get_auth0_audience() -> str:
    return os.getenv("COGNEE_AUTH0_AUDIENCE", DEFAULT_AUTH0_AUDIENCE)


async def device_code_login(
    domain: Optional[str] = None,
    client_id: Optional[str] = None,
    audience: Optional[str] = None,
    scope: str = DEFAULT_SCOPE,
) -> TokenResponse:
    """Run the OAuth 2.0 Device Code Flow against Auth0.

    Prints a URL and code to the terminal, then polls until the user
    approves in the browser. Returns the token set.
    """
    domain = domain or _get_auth0_domain()
    client_id = client_id or _get_auth0_client_id()
    audience = audience or _get_auth0_audience()

    base_url = f"https://{domain}"

    async with aiohttp.ClientSession() as session:
        # Step 1: Request device code
        async with session.post(
            f"{base_url}/oauth/device/code",
            data={
                "client_id": client_id,
                "scope": scope,
                "audience": audience,
            },
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Device code request failed ({resp.status}): {body}")
            device_data = await resp.json()

        device_code = device_data["device_code"]
        user_code = device_data["user_code"]
        verification_uri = (
            device_data.get("verification_uri_complete") or device_data["verification_uri"]
        )
        expires_in = device_data.get("expires_in", 900)
        interval = device_data.get("interval", 5)

        # Step 2: Show the user what to do
        print()
        print("  To authenticate with Cognee Cloud, open this URL in your browser:")
        print()
        print(f"    {verification_uri}")
        print()
        if "verification_uri_complete" not in device_data:
            print(f"  Then enter code: {user_code}")
            print()
        print("  Waiting for authorization...")

        # Step 3: Poll for token
        deadline = time.time() + expires_in
        while time.time() < deadline:
            await asyncio.sleep(interval)

            async with session.post(
                f"{base_url}/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": client_id,
                },
            ) as resp:
                body = await resp.json()

                if resp.status == 200:
                    print("  Authenticated successfully!")
                    return TokenResponse(
                        access_token=body["access_token"],
                        refresh_token=body.get("refresh_token"),
                        id_token=body.get("id_token"),
                        token_type=body.get("token_type", "Bearer"),
                        expires_in=body.get("expires_in", 3600),
                    )

                error = body.get("error", "")
                if error == "authorization_pending":
                    continue
                elif error == "slow_down":
                    interval = min(interval + 5, 30)
                    continue
                elif error == "expired_token":
                    raise TimeoutError("Device code expired. Please try again.")
                elif error == "access_denied":
                    raise PermissionError("Authorization was denied by the user.")
                else:
                    raise RuntimeError(f"Token polling error: {body}")

        raise TimeoutError("Device code flow timed out. Please try again.")


async def refresh_access_token(
    refresh_token: str,
    domain: Optional[str] = None,
    client_id: Optional[str] = None,
) -> TokenResponse:
    """Refresh an expired access token using a refresh token."""
    domain = domain or _get_auth0_domain()
    client_id = client_id or _get_auth0_client_id()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://{domain}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": refresh_token,
            },
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Token refresh failed ({resp.status}): {body}")
            body = await resp.json()
            return TokenResponse(
                access_token=body["access_token"],
                refresh_token=body.get("refresh_token", refresh_token),
                id_token=body.get("id_token"),
                token_type=body.get("token_type", "Bearer"),
                expires_in=body.get("expires_in", 3600),
            )


def extract_email_from_id_token(id_token: str) -> Optional[str]:
    """Decode the JWT payload (without verification) to extract the email claim."""
    import base64
    import json

    try:
        payload_b64 = id_token.split(".")[1]
        # Add padding
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("email")
    except Exception:
        return None
