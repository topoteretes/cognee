import os
from functools import lru_cache
from fastapi_users import models

from fastapi_users.authentication import (
    JWTStrategy,
    AuthenticationBackend,
)

from cognee.shared.auth_secrets import resolve_secret

from .api_bearer import api_bearer_transport, APIJWTStrategy


@lru_cache
def get_api_auth_backend():
    transport = api_bearer_transport

    def get_jwt_strategy() -> JWTStrategy[models.UP, models.ID]:
        secret = resolve_secret("FASTAPI_USERS_JWT_SECRET")
        lifetime_seconds = int(os.getenv("JWT_LIFETIME_SECONDS", "3600"))

        return APIJWTStrategy(secret, lifetime_seconds=lifetime_seconds)

    auth_backend = AuthenticationBackend(
        name=transport.name,
        transport=transport,
        get_strategy=get_jwt_strategy,
    )

    return auth_backend
