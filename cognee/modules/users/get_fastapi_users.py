import uuid
from functools import lru_cache
from fastapi_users import FastAPIUsers

from .authentication.get_api_auth_backend import get_api_auth_backend
from .authentication.get_client_auth_backend import get_client_auth_backend

from .get_user_manager import get_user_manager
from .models.User import User


@lru_cache
def get_fastapi_users():
    api_auth_backend = get_api_auth_backend()
    client_auth_backend = get_client_auth_backend()

    fastapi_users = FastAPIUsers[User, uuid.UUID](
        get_user_manager, [api_auth_backend, client_auth_backend]
    )

    return fastapi_users
