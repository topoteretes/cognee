import os
# from typing import Optional
from functools import lru_cache
from fastapi_users import models
# from fastapi_users.jwt import generate_jwt
from fastapi_users.authentication import (
    JWTStrategy,
    AuthenticationBackend,
)


# from cognee.modules.users.models import User
# from cognee.modules.users.methods import get_user

from .default import default_transport


# class CustomJWTStrategy(JWTStrategy):
#     async def write_token(self, user: User, lifetime_seconds: Optional[int] = None) -> str:
#         # JoinLoad tenant and role information to user object
#         user = await get_user(user.id)

#         data = {"user_id": str(user.id)}

#         return generate_jwt(data, self.encode_key, self.lifetime_seconds, algorithm=self.algorithm)


@lru_cache
def get_auth_backend():
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
