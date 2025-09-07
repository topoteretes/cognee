from fastapi import Depends

from cognee.modules.users.get_fastapi_users import get_fastapi_users
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.authentication.get_client_auth_backend import get_client_auth_backend


def get_auth_router():
    auth_backend = get_client_auth_backend()
    auth_router = get_fastapi_users().get_auth_router(auth_backend)

    @auth_router.get("/me")
    async def get_me(user: User = Depends(get_authenticated_user)):
        return {
            "email": user.email,
        }

    return auth_router
