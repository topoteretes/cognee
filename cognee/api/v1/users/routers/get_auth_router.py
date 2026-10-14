from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordRequestForm

from cognee.modules.users.authentication.default.default_transport import (
    cookie_secure_for,
    default_transport,
)
from cognee.modules.users.authentication.get_client_auth_backend import get_client_auth_backend
from cognee.modules.users.authentication.methods.authenticate_user import authenticate_user
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User


def get_auth_router():
    router = APIRouter()

    @router.post("/login")
    async def login(
        request: Request,
        response: Response,
        credentials: OAuth2PasswordRequestForm = Depends(),
    ):
        """Login — POST /api/v1/auth/login."""
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
            secure=cookie_secure_for(request),
            httponly=default_transport.cookie_httponly,
            samesite=default_transport.cookie_samesite,
        )

        return {"access_token": token, "token_type": "bearer"}

    @router.post("/logout")
    async def logout(
        request: Request, response: Response, user: User = Depends(get_authenticated_user)
    ):
        """Logout — POST /api/v1/auth/logout."""
        # A deletion is itself a Set-Cookie with an expiry in the past; browsers
        # match it against the stored cookie by name, path and domain, and some
        # drop it when Secure or SameSite disagree, so it mirrors the login.
        response.delete_cookie(
            key=default_transport.cookie_name,
            path=default_transport.cookie_path,
            domain=default_transport.cookie_domain,
            secure=cookie_secure_for(request),
            httponly=default_transport.cookie_httponly,
            samesite=default_transport.cookie_samesite,
        )

        return {}

    @router.get("/me")
    async def get_me(user: User = Depends(get_authenticated_user)):
        """Get me — GET /api/v1/auth/me."""
        return {
            "email": user.email,
        }

    return router
