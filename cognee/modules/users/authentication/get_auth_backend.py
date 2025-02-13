import os
from functools import lru_cache
from fastapi_users import models
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from datetime import datetime, timedelta
from typing import Optional


class CustomJWTStrategy(JWTStrategy):
    def create_access_token(
        self, subject: str, tenant: str, role: str, lifetime_seconds: Optional[int] = None
    ) -> str:
        lifetime = (
            timedelta(seconds=lifetime_seconds) if lifetime_seconds else self.lifetime_seconds
        )
        expire = datetime.utcnow() + lifetime
        to_encode = {"sub": subject, "exp": expire, "tenant": tenant, "role": role}

        return self.encode(to_encode)


@lru_cache
def get_auth_backend():
    bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

    def get_jwt_strategy() -> JWTStrategy[models.UP, models.ID]:
        secret = os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret")
        return CustomJWTStrategy(secret, lifetime_seconds=3600)

    auth_backend = AuthenticationBackend(
        name="jwt",
        transport=bearer_transport,
        get_strategy=get_jwt_strategy,
    )

    return auth_backend
