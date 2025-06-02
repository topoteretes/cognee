from fastapi_users.authentication import CookieTransport

from .auth0_config import get_auth0_config


auth0_transport = CookieTransport(
    cookie_name=get_auth0_config().auth_token_cookie_name,
    cookie_httponly=True,
    cookie_samesite="Lax",
)

auth0_transport.name = "cookie"
