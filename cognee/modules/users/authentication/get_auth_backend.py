import os
from functools import lru_cache
from fastapi_users import models
from fastapi_users.authentication import AuthenticationBackend, JWTStrategy

from .default import default_transport


@lru_cache
def get_auth_backend():
    transport = default_transport

    if os.getenv("USE_AUTH0_AUTHORIZATION") == "True":
        from .auth0 import auth0_transport

        transport = auth0_transport

    def get_jwt_strategy() -> JWTStrategy[models.UP, models.ID]:
        if os.getenv("USE_AUTH0_AUTHORIZATION") == "True":
            from .auth0 import Auth0JWTStrategy

            return Auth0JWTStrategy(secret="NOT IMPORTANT FOR AUTH0", lifetime_seconds=36000) # 10 hours is default token lifetime
        else:
            from .default.default_jwt_strategy import DefaultJWTStrategy

            secret = os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret")

            return DefaultJWTStrategy(secret, lifetime_seconds=3600)

    auth_backend = AuthenticationBackend(
        name=transport.name,
        transport=transport,
        get_strategy=get_jwt_strategy,
    )

    return auth_backend
