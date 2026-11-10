import os

from fastapi_users.authentication import CookieTransport

# Get cookie domain from environment variable
# If not set or empty, use None to allow cookie to work on any domain
cookie_domain = os.getenv("AUTH_TOKEN_COOKIE_DOMAIN")
if cookie_domain == "":
    cookie_domain = None


def _secure_cookie_from_env() -> bool:
    """Whether the auth cookie gets Secure, from AUTH_TOKEN_COOKIE_SECURE.

    Accepts true or false (default false); anything else raises.
    """
    value = os.getenv("AUTH_TOKEN_COOKIE_SECURE", "false").strip().lower()
    if value not in ("true", "false"):
        raise ValueError(f"AUTH_TOKEN_COOKIE_SECURE must be true or false, got {value!r}")
    return value == "true"


# Note: no cookie_max_age is set, so the cookie lasts for the browser session. JWT_LIFETIME_SECONDS
# (get_client_auth_backend.py) only limits how long the token inside it is accepted.

default_transport = CookieTransport(
    cookie_name=os.getenv("AUTH_TOKEN_COOKIE_NAME", "auth_token"),
    # Secure: the browser then sends the cookie over https only. Off by default
    # because a browser will not store a Secure cookie from http://localhost;
    # set AUTH_TOKEN_COOKIE_SECURE=true behind TLS (cognee itself sees plain
    # http behind a terminating proxy, so it cannot infer this).
    cookie_secure=_secure_cookie_from_env(),
    cookie_httponly=True,
    cookie_samesite="Lax",
    cookie_domain=cookie_domain,  # None allows cookie to work on any domain
)

default_transport.name = "cookie"
