import os
from functools import lru_cache
from fastapi_users import models

from fastapi_users.authentication import (
    JWTStrategy,
    AuthenticationBackend,
)

from .default import default_transport


@lru_cache
def get_client_auth_backend():
    transport = default_transport

    def get_jwt_strategy() -> JWTStrategy[models.UP, models.ID]:
        from .default.default_jwt_strategy import DefaultJWTStrategy

        secret = os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret")

        return DefaultJWTStrategy(secret, lifetime_seconds=3600)

    auth_backend = AuthenticationBackend(
        name=transport.name,
        transport=transport,
        get_strategy=get_jwt_strategy,
    )

    return auth_backend
