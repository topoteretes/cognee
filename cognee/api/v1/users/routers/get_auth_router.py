from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordRequestForm

from cognee.modules.users.authentication.default.default_transport import default_transport
from cognee.modules.users.authentication.get_client_auth_backend import get_client_auth_backend
from cognee.modules.users.authentication.methods.authenticate_user import authenticate_user
from cognee.modules.users.authentication.refresh_token import (
    create_refresh_token,
    delete_refresh_token,
    get_refresh_token_lifetime,
    redeem_refresh_token,
)
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User


def _refresh_cookie_name() -> str:
    return f"{default_transport.cookie_name}_refresh"


def _refresh_cookie_path(request: Request) -> str:
    """The path login, refresh and logout share, as this request addressed it.

    The refresh cookie is limited to it, so the browser sends the long-lived
    token to these routes only and not with every API request.
    """
    return request.url.path.rsplit("/", 1)[0]


def _set_cookie(response: Response, name: str, value: str, max_age: int | None, path: str) -> None:
    response.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        path=path,
        domain=default_transport.cookie_domain,
        secure=default_transport.cookie_secure,
        httponly=default_transport.cookie_httponly,
        samesite=default_transport.cookie_samesite,
    )


def _delete_cookie(response: Response, name: str, path: str) -> None:
    # Same attributes as _set_cookie: a deletion is a Set-Cookie with a past
    # expiry, and a browser only drops the stored cookie when its name, path
    # and domain match.
    response.delete_cookie(
        key=name,
        path=path,
        domain=default_transport.cookie_domain,
        secure=default_transport.cookie_secure,
        httponly=default_transport.cookie_httponly,
        samesite=default_transport.cookie_samesite,
    )


async def _write_session(
    request: Request, response: Response, user: User, refresh_token: str
) -> dict[str, str]:
    """Set a new access token and the refresh token as cookies; return the token body.

    The refresh cookie is written on every call, so the browser keeps it for as
    long as the server keeps the token: its age restarts with each use.
    """
    token = await get_client_auth_backend().get_strategy().write_token(user)

    _set_cookie(
        response,
        default_transport.cookie_name,
        token,
        default_transport.cookie_max_age,
        default_transport.cookie_path,
    )
    _set_cookie(
        response,
        _refresh_cookie_name(),
        refresh_token,
        int(get_refresh_token_lifetime().total_seconds()),
        _refresh_cookie_path(request),
    )

    return {"access_token": token, "token_type": "bearer"}


def get_auth_router():
    router = APIRouter()

    @router.post("/login")
    async def login(
        request: Request,
        response: Response,
        credentials: OAuth2PasswordRequestForm = Depends(),
    ):
        """Exchange email and password for an access token.

        Takes username and password as form fields. Returns access_token and token_type, and
        sets the same token as the authentication cookie. It also sets a refresh cookie, which
        POST /refresh exchanges for a new access token once this one expires; the browser
        sends that cookie to the login, refresh and logout routes only. Unknown
        credentials, a wrong password or an inactive user return 400 with detail
        LOGIN_BAD_CREDENTIALS, as does a malformed form body. An account without a password,
        such as an agent, returns 400 and must use API key authentication instead.
        """
        user = await authenticate_user(credentials.username, credentials.password)

        if user is None:
            raise HTTPException(status_code=400, detail="LOGIN_BAD_CREDENTIALS")

        return await _write_session(request, response, user, await create_refresh_token(user))

    @router.post("/refresh")
    async def refresh(request: Request, response: Response):
        """Exchange the refresh cookie for a new access token.

        Takes no body: the refresh token is read from the cookie login set. Returns
        access_token and token_type, sets the new token as the authentication cookie, and
        extends the refresh token by REFRESH_TOKEN_LIFETIME_SECONDS. A missing, unknown or
        expired refresh token returns 401 with detail REFRESH_BAD_TOKEN, as does one whose
        user is inactive or has changed their password since logging in, and one issued
        under another FASTAPI_USERS_JWT_SECRET.
        """
        refresh_token = request.cookies.get(_refresh_cookie_name(), "")
        user = await redeem_refresh_token(refresh_token)

        if user is None:
            raise HTTPException(status_code=401, detail="REFRESH_BAD_TOKEN")

        return await _write_session(request, response, user, refresh_token)

    @router.post("/logout")
    async def logout(request: Request, response: Response):
        """End the login of the calling browser.

        Deletes the refresh token named by the refresh cookie and clears the authentication
        and refresh cookies, with the name, path and domain used at login. Returns an empty
        JSON object. It needs no valid access token, since a login must also be endable after
        that has expired. The access token itself is not otherwise invalidated: a copy of it
        works until it expires.
        """
        refresh_token = request.cookies.get(_refresh_cookie_name())
        if refresh_token:
            await delete_refresh_token(refresh_token)

        _delete_cookie(response, default_transport.cookie_name, default_transport.cookie_path)
        _delete_cookie(response, _refresh_cookie_name(), _refresh_cookie_path(request))

        return {}

    @router.get("/me")
    async def get_me(user: User = Depends(get_authenticated_user)):
        """Return the authenticated user's email address.

        The response body contains only the email field.
        """
        return {
            "email": user.email,
        }

    return router
