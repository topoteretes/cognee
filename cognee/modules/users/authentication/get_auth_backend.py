import os
from functools import lru_cache
from fastapi_users import models
from fastapi_users.jwt import generate_jwt
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)

from typing import Optional

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_user


class CustomJWTStrategy(JWTStrategy):
    async def write_token(self, user: User, lifetime_seconds: Optional[int] = None) -> str:
        # JoinLoad tenant and role information to user object
        user = await get_user(user.id)

        data = {"user_id": str(user.id)}

        return generate_jwt(data, self.encode_key, self.lifetime_seconds, algorithm=self.algorithm)


@lru_cache
def get_auth_backend():
    bearer_transport = BearerTransport(tokenUrl="api/v1/auth/login")

    def get_jwt_strategy() -> JWTStrategy[models.UP, models.ID]:
        secret = os.getenv("FASTAPI_USERS_JWT_SECRET")
        if not secret:
            raise RuntimeError(
                "FASTAPI_USERS_JWT_SECRET environment variable must be set and non-empty for JWT authentication."
            )
        return CustomJWTStrategy(secret, lifetime_seconds=3600)

    auth_backend = AuthenticationBackend(
        name="jwt",
        transport=bearer_transport,
        get_strategy=get_jwt_strategy,
    )

    return auth_backend