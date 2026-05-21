from fastapi import Depends, HTTPException, Response, APIRouter
from fastapi.security import OAuth2PasswordRequestForm

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.authentication.get_client_auth_backend import get_client_auth_backend
from cognee.modules.users.authentication.methods.authenticate_user import authenticate_user
from cognee.modules.users.authentication.default.default_transport import default_transport


def get_auth_router():
    router = APIRouter()

    @router.post("/login")
    async def login(
        response: Response,
        credentials: OAuth2PasswordRequestForm = Depends(),
    ):
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
        response.delete_cookie(
            key=default_transport.cookie_name,
            domain=default_transport.cookie_domain,
        )

        return {}

    @router.get("/me")
    async def get_me(user: User = Depends(get_authenticated_user)):
        return {
            "email": user.email,
        }

    return router
