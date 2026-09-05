"""API-key bootstrap against a local or self-hosted Cognee server.

``serve(url=...)`` needs an API key to authenticate, but a freshly
started server has none to hand out-of-band. Every agent integration
solves this the same way: log in as the (default) user once, mint an
API key over the JWT session, cache it, and use ``X-Api-Key`` from then
on. This module is that flow, so ``serve()`` can connect to a fresh
server without a pre-minted key.

The JWT is used only for this bootstrap — it authenticates the api-keys
calls via the ``auth_token`` cookie (the server's fastapi-users cookie
transport), matching the flow the integrations already use.
"""

import asyncio
import ipaddress
import os
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from cognee.api.v1.serve.exceptions import (
    CogneeAuthError,
    CogneeTransportError,
    http_error_for_status,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("serve.local_auth")

DEFAULT_BOOTSTRAP_KEY_NAME = "cognee-serve-bootstrap"

_BOOTSTRAP_TIMEOUT = aiohttp.ClientTimeout(total=30, sock_connect=10)


def is_private_host(service_url: str) -> bool:
    """True when the URL points at loopback or a private (RFC-1918) address.

    Only IP literals and ``localhost`` names qualify — a DNS hostname can
    resolve anywhere, so it never counts as private.
    """
    host = urlparse(service_url).hostname or ""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def bootstrap_allowed(service_url: str, override: Optional[bool] = None) -> bool:
    """Decide whether serve() may log in to mint a key for this URL.

    The login flow sends (possibly real) user credentials to the host, so
    it is restricted to hosts the caller plausibly controls: private
    addresses by default, anything else only on explicit opt-in — the
    ``override`` argument (``bootstrap_auth=`` on serve()) or
    ``COGNEE_AUTH_BOOTSTRAP=true``. An explicit ``False`` disables the
    bootstrap even for localhost.
    """
    if override is not None:
        return override
    env_flag = os.getenv("COGNEE_AUTH_BOOTSTRAP", "").strip().lower()
    if env_flag in ("true", "1", "yes"):
        return True
    return is_private_host(service_url)


def resolve_bootstrap_credentials() -> tuple[str, str]:
    """Resolve the login used to mint a key, matching the integrations' chain.

    ``COGNEE_USER_EMAIL``/``COGNEE_USER_PASSWORD`` (the plugins' variables)
    win over ``DEFAULT_USER_EMAIL``/``DEFAULT_USER_PASSWORD`` (the server's
    own default-user overrides); the fallback is the server's built-in
    default user.
    """
    email = (
        os.getenv("COGNEE_USER_EMAIL")
        or os.getenv("DEFAULT_USER_EMAIL")
        or "default_user@example.com"
    )
    password = (
        os.getenv("COGNEE_USER_PASSWORD")
        or os.getenv("DEFAULT_USER_PASSWORD")
        or "default_password"
    )
    return email, password


def _is_usable_key(key: str) -> bool:
    # With HASH_API_KEY enabled the list endpoint masks keys as "************";
    # a masked placeholder must never be treated as a credential.
    return bool(key) and "*" not in key


async def login_and_mint_api_key(
    service_url: str,
    email: Optional[str] = None,
    password: Optional[str] = None,
    key_name: str = DEFAULT_BOOTSTRAP_KEY_NAME,
) -> str:
    """Log in to the instance and return a usable API key.

    Reuses the first usable existing key; otherwise creates one named
    ``key_name``. Raises ``CogneeAuthError`` when the login is rejected
    and ``CogneeTransportError`` when the server is unreachable.
    """
    service_url = service_url.rstrip("/")
    if email is None or password is None:
        default_email, default_password = resolve_bootstrap_credentials()
        email = email or default_email
        password = password or default_password

    try:
        async with aiohttp.ClientSession(timeout=_BOOTSTRAP_TIMEOUT) as session:
            async with session.post(
                f"{service_url}/api/v1/auth/login",
                data={"username": email, "password": password},
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise CogneeAuthError(
                        f"Login failed for {email} ({resp.status}): {body[:200]}. "
                        f"Set COGNEE_USER_EMAIL/COGNEE_USER_PASSWORD to a valid account.",
                        status=resp.status,
                        body=body,
                        operation="login",
                    )
                login_data = await resp.json()

            access_token = str(login_data.get("access_token") or "")
            if not access_token:
                raise CogneeAuthError(
                    "Login returned no access token.",
                    status=200,
                    body=login_data,
                    operation="login",
                )
            cookies = {"auth_token": access_token}

            async with session.get(f"{service_url}/api/v1/auth/api-keys", cookies=cookies) as resp:
                if resp.status == 200:
                    keys = await resp.json()
                    for entry in keys if isinstance(keys, list) else []:
                        key = str(entry.get("key") or "")
                        if _is_usable_key(key):
                            return key

            async with session.post(
                f"{service_url}/api/v1/auth/api-keys",
                json={"name": key_name},
                cookies=cookies,
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise http_error_for_status(resp.status, body, operation="mint_api_key")
                created = await resp.json()

            key = str(created.get("key") or "")
            if not _is_usable_key(key):
                raise CogneeAuthError(
                    "API key creation returned no usable key.",
                    status=200,
                    body=created,
                    operation="mint_api_key",
                )
            return key
    except (aiohttp.ClientError, asyncio.TimeoutError) as error:
        raise CogneeTransportError(
            f"Could not reach the Cognee instance at {service_url} during auth bootstrap: {error}",
            operation="mint_api_key",
            cause=error,
        ) from error
