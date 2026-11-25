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

        Takes username and password as form fields. Returns access_token and token_type, and
        sets the same token as the authentication cookie. Unknown credentials, a wrong
        password or an inactive user return 400 with detail LOGIN_BAD_CREDENTIALS, as does a
        malformed form body. An account without a password, such as an agent, returns 400
        and must use API key authentication instead.
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
        """Clear the authentication cookie for the current user.

        Requires authentication and returns an empty JSON object. It deletes the cookie with
        the same name, path and domain used at login; the token itself is not otherwise
        invalidated.
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
        """Return the authenticated user's email address.

        The response body contains only the email field.
        """
        return {
            "email": user.email,
        }

    return router
