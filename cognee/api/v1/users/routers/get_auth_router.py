from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.security import OAuth2PasswordRequestForm

from cognee.modules.users.authentication.default.default_transport import default_transport
from cognee.modules.users.authentication.get_client_auth_backend import get_client_auth_backend
from cognee.modules.users.authentication.methods.authenticate_user import authenticate_user
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User


def get_auth_router():
    router = APIRouter()

    @router.post("/login")
    async def login(
        response: Response,
        credentials: OAuth2PasswordRequestForm = Depends(),
    ):
        """Exchange email and password for an access token.

        Takes an OAuth2 password form body; a malformed body, wrong credentials or an
        inactive user all return 400 with detail LOGIN_BAD_CREDENTIALS. On success the JWT
        is returned as access_token with token_type bearer and is also set as the
        authentication cookie.
        """
        user = await authenticate_user(credentials.username, credentials.password)

        if user is None:
            raise HTTPException(status_code=400, detail="LOGIN_BAD_CREDENTIALS")

        client_backend = get_client_auth_backend()
        strategy = client_backend.get_strategy()
        token = await strategy.write_token(user)

        response.set_cookie(
            key=default_transport.cookie_name,
            value=token,
            max_age=default_transport.cookie_max_age,
            path=default_transport.cookie_path,
            domain=default_transport.cookie_domain,
            secure=default_transport.cookie_secure,
            httponly=default_transport.cookie_httponly,
            samesite=default_transport.cookie_samesite,
        )

        return {"access_token": token, "token_type": "bearer"}

    @router.post("/logout")
    async def logout(response: Response, user: User = Depends(get_authenticated_user)):
        """Clear the authentication cookie.

        Requires an authenticated caller and re-sends the login cookie with the same name,
        path and domain so the browser drops it. The response body is an empty JSON object.
        """
        # Same attributes as the login's set_cookie: a deletion is a Set-Cookie
        # with a past expiry, and a browser only drops the stored cookie when
        # its name, path and domain match.
        response.delete_cookie(
            key=default_transport.cookie_name,
            path=default_transport.cookie_path,
            domain=default_transport.cookie_domain,
            secure=default_transport.cookie_secure,
            httponly=default_transport.cookie_httponly,
            samesite=default_transport.cookie_samesite,
        )

        return {}

    @router.get("/me")
    async def get_me(user: User = Depends(get_authenticated_user)):
        """Return the authenticated user's email.

        The response body has a single email field for the caller resolved from the token or
        cookie.
        """
        return {
            "email": user.email,
        }

    return router
