import os

from fastapi import Request
from fastapi_users.authentication import CookieTransport

# Get cookie domain from environment variable
# If not set or empty, use None to allow cookie to work on any domain
cookie_domain = os.getenv("AUTH_TOKEN_COOKIE_DOMAIN")
if cookie_domain == "":
    cookie_domain = None


def cookie_secure_setting() -> bool | None:
    """``AUTH_TOKEN_COOKIE_SECURE`` as a tri-state: True, False, or None for auto.

    The auth cookie is the session: a browser sends it on every request, and
    the ``Secure`` attribute is what stops it from sending the cookie over
    plain http, where anyone on the path can read the token. Behind a
    TLS-terminating proxy cognee itself only ever sees http, so the attribute
    cannot be inferred from the socket; ``true`` sets it unconditionally and
    is the setting for anything behind TLS. A browser refuses to store a
    ``Secure`` cookie from ``http://localhost``, so ``false`` keeps a plain
    local deployment logging in. Unset (auto) follows the scheme of the
    request as cognee sees it: https only when cognee serves TLS itself or
    the server trusts the proxy's X-Forwarded-Proto, which uvicorn and
    gunicorn do only for addresses in FORWARDED_ALLOW_IPS (default: loopback).

    Read on every call (one getenv per login) so the setting can be changed
    without re-importing, in tests above all.
    """
    raw = os.getenv("AUTH_TOKEN_COOKIE_SECURE", "").strip().lower()
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    return None


def cookie_secure_for(request: Request) -> bool:
    """Whether the auth cookie written for ``request`` carries ``Secure``."""
    setting = cookie_secure_setting()
    if setting is not None:
        return setting
    return request.url.scheme == "https"


# Note: Cookie expiration is automatically set by FastAPI Users based on JWT Strategy's lifetime_seconds
# The JWT Strategy lifetime_seconds is configured in get_client_auth_backend.py
# and reads from JWT_LIFETIME_SECONDS environment variable

default_transport = CookieTransport(
    cookie_name=os.getenv("AUTH_TOKEN_COOKIE_NAME", "auth_token"),
    # Not what the cookie gets: cognee's own login and logout routes resolve
    # Secure per request with cookie_secure_for(). This backs only
    # fastapi-users' transport methods, which cognee does not mount.
    cookie_secure=False,
    cookie_httponly=True,
    cookie_samesite="Lax",
    cookie_domain=cookie_domain,  # None allows cookie to work on any domain
)

default_transport.name = "cookie"
