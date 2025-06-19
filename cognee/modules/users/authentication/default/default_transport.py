import os
from fastapi_users.authentication import CookieTransport

default_transport = CookieTransport(
    cookie_name=os.getenv("AUTH_TOKEN_COOKIE_NAME", "auth_token"),
    cookie_secure=False,
    cookie_httponly=True,
    cookie_samesite="Lax",
    cookie_domain="localhost",
)

default_transport.name = "cookie"
