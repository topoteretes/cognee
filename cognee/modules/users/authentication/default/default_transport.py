import os
from fastapi_users.authentication import CookieTransport

# Get cookie domain from environment variable
# If not set or empty, use None to allow cookie to work on any domain
cookie_domain = os.getenv("AUTH_TOKEN_COOKIE_DOMAIN")
if cookie_domain == "":
    cookie_domain = None

# Note: Cookie expiration is automatically set by FastAPI Users based on JWT Strategy's lifetime_seconds
# The JWT Strategy lifetime_seconds is configured in get_client_auth_backend.py
# and reads from JWT_LIFETIME_SECONDS environment variable

default_transport = CookieTransport(
    cookie_name=os.getenv("AUTH_TOKEN_COOKIE_NAME", "auth_token"),
    cookie_secure=False,
    cookie_httponly=True,
    cookie_samesite="Lax",
    cookie_domain=cookie_domain,  # None allows cookie to work on any domain
)

default_transport.name = "cookie"
