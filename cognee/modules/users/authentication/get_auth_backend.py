import os
from functools import lru_cache
from fastapi_users import models
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)


@lru_cache
def get_auth_backend():
    bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

    def get_jwt_strategy() -> JWTStrategy[models.UP, models.ID]:
        secret = os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret")
        return JWTStrategy(secret, lifetime_seconds=3600)

    auth_backend = AuthenticationBackend(
        name="jwt",
        transport=bearer_transport,
        get_strategy=get_jwt_strategy,
    )

    return auth_backend
