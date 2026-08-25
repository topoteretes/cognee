"""GitHub App authentication: app JWT and short-lived installation tokens.

A GitHub App holds no durable per-installation secret — the durable secret
is the app's RS256 private key (env), from which a ~9-minute JWT is signed,
which in turn mints a ~1-hour installation access token scoped to one
installation's repositories
(https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app).
That's why the stored ``IntegrationCredential`` for GitHub carries an empty
token payload: tokens are minted here on demand, never persisted.

The JWT is built by hand with ``cryptography`` (already a core dependency)
rather than pulling in a JWT library for one fixed-shape, fixed-alg token.
"""

import base64
import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from cognee.modules.integrations.github.github_settings import private_key_pem, require

API_BASE_URL = "https://api.github.com"

_TIMEOUT = aiohttp.ClientTimeout(total=30)

# GitHub caps app JWTs at 10 minutes; 9 leaves clock-skew headroom on top of
# the 60-second iat backdate GitHub itself recommends.
_JWT_BACKDATE_SECONDS = 60
_JWT_LIFETIME_SECONDS = 9 * 60


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def build_app_jwt(*, now: Optional[int] = None) -> str:
    """Sign a short-lived RS256 JWT identifying the app itself.

    ``now`` is injectable for tests; production callers omit it.
    """
    issued_at = int(now if now is not None else time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    payload = {
        "iat": issued_at - _JWT_BACKDATE_SECONDS,
        "exp": issued_at + _JWT_LIFETIME_SECONDS,
        "iss": require("app_id"),
    }
    signing_input = (
        f"{_b64url(json.dumps(header, separators=(',', ':')).encode())}"
        f".{_b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    )
    private_key = serialization.load_pem_private_key(private_key_pem().encode(), password=None)
    if not isinstance(private_key, RSAPrivateKey):
        raise RuntimeError("GITHUB_APP_PRIVATE_KEY is not an RSA private key")
    signature = private_key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input}.{_b64url(signature)}"


def _api_headers(bearer: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {bearer}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def get_installation(installation_id: int) -> dict[str, Any]:
    """Fetch one installation of *this* app, authenticated as the app.

    The authoritative source for install metadata (org login, repository
    selection): querying it with the app JWT proves the installation id
    belongs to our app, so nothing from an unauthenticated callback's query
    string is ever trusted directly.

    Raises ``RuntimeError`` when GitHub rejects the lookup (unknown id,
    suspended installation, bad app credentials).
    """
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        async with session.get(
            f"{API_BASE_URL}/app/installations/{installation_id}",
            headers=_api_headers(build_app_jwt()),
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"GitHub installation lookup failed for {installation_id}: "
                    f"HTTP {response.status}"
                )
            return await response.json()


async def mint_installation_token(installation_id: int) -> tuple[str, Optional[datetime]]:
    """Mint a fresh installation access token (valid ~1 hour).

    Returns ``(token, expires_at)``. Callers use the token immediately and
    discard it — it is never written to the credential store.
    """
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        async with session.post(
            f"{API_BASE_URL}/app/installations/{installation_id}/access_tokens",
            headers=_api_headers(build_app_jwt()),
        ) as response:
            if response.status != 201:
                raise RuntimeError(
                    f"GitHub installation token mint failed for {installation_id}: "
                    f"HTTP {response.status}"
                )
            payload = await response.json()

    expires_at = None
    expires_raw = payload.get("expires_at")
    if expires_raw:
        expires_at = datetime.fromisoformat(expires_raw.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    return payload["token"], expires_at


async def user_can_access_installation(user_token: str, installation_id: int) -> bool:
    """Whether the GitHub user behind ``user_token`` can access the installation.

    The install callback is unauthenticated and ``installation_id`` arrives
    in its query string, where anyone can put any (small, guessable) integer.
    This check — GitHub's documented mitigation — confirms the person who
    completed the OAuth leg actually has access to that installation, so a
    user can't bind another org's installation to their own cognee account.
    """
    url: Optional[str] = f"{API_BASE_URL}/user/installations?per_page=100"
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        while url:
            async with session.get(url, headers=_api_headers(user_token)) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"GitHub user-installations lookup failed: HTTP {response.status}"
                    )
                payload = await response.json()
                for installation in payload.get("installations", []):
                    if installation.get("id") == installation_id:
                        return True
                next_link = response.links.get("next", {}).get("url")
                url = str(next_link) if next_link else None
    return False
